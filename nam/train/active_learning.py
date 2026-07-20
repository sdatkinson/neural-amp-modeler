"""
Active-learning ensemble training adapted from PANAMA.

The serial, device-agnostic ensemble flow follows PANAMA (Parametric Active-learning
for Neural Amp Modeling Assistance), arXiv:2509.26564v1, adapted to this repo's
parametric NAM training stack and runtime-selected device handling.
"""

import contextlib as _contextlib
import gc as _gc
import importlib as _importlib
import json as _json
import math as _math
import os as _os
import shutil as _shutil
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from concurrent.futures import as_completed as _as_completed
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from collections.abc import Sequence as _Sequence
from typing import Any as _Any
from typing import cast as _cast
from warnings import warn as _warn

# CUDA LSTM kernels need a cuBLAS workspace configuration for deterministic execution.
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

import pytorch_lightning as _pl
import torch as _torch
import torch.multiprocessing as _torch_mp
from tqdm import tqdm as _tqdm
from lightning_fabric.utilities.warnings import PossibleUserWarning as _PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import TensorDataset as _TensorDataset

from nam.data import ConcatDataset as _ConcatDataset
from nam.data import Split as _Split
from nam.data import WavInfo as _WavInfo
from nam.data import apply_joint_dataset_hooks as _apply_joint_dataset_hooks
from nam.data import get_joint_dataset_hooks as _get_joint_dataset_hooks
from nam.data import init_dataset as _init_dataset
from nam.data import wav_to_tensor as _wav_to_tensor
from nam.models.parametric import abbreviate_param_names as _abbreviate_param_names
from nam.models.parametric import assemble_raw_params as _assemble_raw_params
from nam.models.parametric import data_config_from_model as _data_config_from_model
from nam.models.parametric import decode_named_params as _decode_named_params
from nam.models.parametric import make_capture_y_path as _make_capture_y_path
from nam.models.parametric import quantize_to_capture_grid as _quantize_to_capture_grid
from nam.models.parametric import split_param_indices as _split_param_indices
from nam.models.parametric import switch_combinations as _switch_combinations
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric._dataset import _coerce_param_specs
from nam.train.full import _handshake_datasets
from nam.train.parametric import _MelSpectrogram
from nam.train.parametric import _ParametricLightningModule
from nam.train.parametric import _create_parametric_callbacks
from nam.train.parametric import _iter_inner_datasets
from nam.train.parametric import _make_parametric_dataloader
from nam.util import filter_warnings as _filter_warnings

_torch.set_float32_matmul_precision("medium")

__all__ = [
    "DisagreementCandidate",
    "RoundResult",
    "append_to_data_config",
    "cluster_and_select",
    "emit_proposals",
    "find_disagreement_settings",
    "run_round",
    "train_ensemble",
]

_DEFAULT_CLUSTER_THRESHOLD = 0.1
_QUANTIZED_DEDUPE_DECIMALS = 8
# Only these (windowing) keys are inherited from an existing train entry when appending a
# proposed capture: the new captures should share the project's train windowing without
# dragging along incidental per-entry keys (path overrides, comments, ...).
_INHERITED_TRAIN_WINDOW_KEYS = ("start_seconds", "stop_seconds", "ny")


@_dataclass(frozen=True)
class DisagreementCandidate:
    raw_params: _torch.Tensor
    switch_combo: tuple[int, ...]
    score: float


def _resolve_device() -> _torch.device:
    if _torch.cuda.is_available():
        return _torch.device("cuda")
    mps_backend = getattr(_torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return _torch.device("mps")
    return _torch.device("cpu")


def _trainer_device_config(device: _torch.device) -> dict[str, str | int | list[int]]:
    if device.type == "cuda":
        # Pin the exact GPU when an index is set (parallel members are round-robined
        # across GPUs as cuda:idx); devices=[idx] targets that card, whereas devices=1
        # would let Lightning pick cuda:0 for every worker.
        if device.index is not None:
            return {"accelerator": "gpu", "devices": [device.index]}
        return {"accelerator": "gpu", "devices": 1}
    if device.type == "mps":
        return {"accelerator": "mps", "devices": 1}
    if device.type == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    raise ValueError(f"Unsupported device type {device.type!r}")


def _prepare_learning_config(
    learning_config: dict,
    device: _torch.device,
) -> dict:
    learning_config = _deepcopy(learning_config)
    trainer_config = dict(learning_config["trainer"])
    trainer_config.update(_trainer_device_config(device))
    learning_config["trainer"] = trainer_config
    return learning_config


def _device_memory_bytes(device: _torch.device) -> tuple[int | None, int | None]:
    if device.type == "cuda":
        with _torch.cuda.device(device):
            free, total = _torch.cuda.mem_get_info()
        return int(free), int(total)
    if device.type == "mps":
        mps = getattr(_torch, "mps", None)
        if mps is not None and hasattr(mps, "recommended_max_memory"):
            total = int(mps.recommended_max_memory())
            free = total - int(mps.current_allocated_memory())
            return max(free, 0), total
    return None, None


def _closest_effective_batch_plan(
    *, physical_ceiling: int, target: int
) -> tuple[int, int, int]:
    physical_ceiling = max(physical_ceiling, 1)
    target = max(target, 1)
    best: tuple[int, int, int, int] | None = None
    tolerance = max(round(target * 0.02), 1)
    for accumulation in range(1, target + 1):
        physical = min(physical_ceiling, max(round(target / accumulation), 1))
        effective = physical * accumulation
        error = abs(effective - target)
        candidate = (
            0 if error <= tolerance else 1,
            -physical if error <= tolerance else error,
            error,
            accumulation,
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("Failed to resolve an active-learning batch plan")
    _, _, _, accumulation = best
    physical = min(
        physical_ceiling, max(round(target / accumulation), 1)
    )
    return physical, accumulation, physical * accumulation


def _resolve_member_batching(
    learning_config: dict,
    *,
    device: _torch.device,
    num_train_windows: int,
    num_validation_windows: int,
) -> dict[str, _Any]:
    sizing = learning_config.get("batch_sizing", {})
    train_config = learning_config["train_dataloader"]
    trainer_config = learning_config["trainer"]
    free_bytes, total_bytes = _device_memory_bytes(device)

    if sizing.get("auto") and free_bytes is not None:
        memory_fraction = float(sizing["memory_fraction"])
        bytes_per_window = int(sizing["bytes_per_timestep"]) * int(sizing["ny"])
        memory_ceiling = max(int(free_bytes * memory_fraction) // bytes_per_window, 1)
        physical_ceiling = min(
            memory_ceiling,
            int(sizing["max_physical_batch_size"]),
            max(num_train_windows, 1),
        )
        target = min(
            int(sizing["target_effective_batch_size"]),
            max(num_train_windows, 1),
        )
        physical, accumulation, effective = _closest_effective_batch_plan(
            physical_ceiling=physical_ceiling,
            target=target,
        )
        train_config["batch_size"] = physical
        train_config["drop_last"] = True
        trainer_config["accumulate_grad_batches"] = accumulation
        learning_config["val_dataloader"]["batch_size"] = min(
            memory_ceiling,
            int(sizing["max_physical_batch_size"]),
            max(num_validation_windows, 1),
        )
    else:
        physical = int(train_config["batch_size"])
        accumulation = int(trainer_config.get("accumulate_grad_batches", 1))
        effective = physical * accumulation
        memory_ceiling = None

    return {
        "device": str(device),
        "device_name": (
            _torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
        ),
        "free_vram_bytes": free_bytes,
        "total_vram_bytes": total_bytes,
        "memory_batch_ceiling": memory_ceiling,
        "physical_batch_size": physical,
        "accumulate_grad_batches": accumulation,
        "effective_batch_size": effective,
        "target_effective_batch_size": sizing.get("target_effective_batch_size"),
        "train_windows": num_train_windows,
        "validation_windows": num_validation_windows,
    }


def _canonical_param_specs(raw_param_specs: _Any) -> list[dict[str, _Any]]:
    # Reuse the stock coercion/validation (unique names, ParamSpec-or-mapping) and
    # canonicalize to dicts so two spec lists can be compared by value.
    return [spec.to_dict() for spec in _coerce_param_specs(raw_param_specs)]


def _model_param_specs(model_config: dict) -> list[dict[str, _Any]]:
    try:
        return _canonical_param_specs(model_config["net"]["config"]["params"])
    except KeyError as exc:
        raise ValueError(
            "Model config must define net.config.params for parametric dataset loading"
        ) from exc


def _validate_existing_param_specs_match(
    common: dict,
    model_param_specs: list[dict[str, _Any]],
) -> None:
    # No-op when common carries no param_specs; otherwise it must match the model's.
    if (
        "param_specs" in common
        and _canonical_param_specs(common["param_specs"]) != model_param_specs
    ):
        raise ValueError(
            "Data config common.param_specs does not match "
            "model_config['net']['config']['params']"
        )


def _prepare_data_config(data_config: dict, model_config: dict) -> dict:
    data_config = _deepcopy(data_config)
    common = data_config.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("Data config common section must be a mapping")

    # Aggregated configs re-fed across rounds (Task 7, round > 0) may already carry
    # param_specs. Tolerate that only when it matches the model's specs, then strip it
    # so data_config_from_model re-injects the canonical copy (it raises on duplicates).
    if "param_specs" in common:
        _validate_existing_param_specs_match(common, _model_param_specs(model_config))
        common = dict(common)
        del common["param_specs"]
        data_config["common"] = common

    return _data_config_from_model(data_config, model_config)


def _prepare_member_data_config(
    data_config: dict,
    receptive_field: int,
) -> dict:
    data_config = _deepcopy(data_config)
    common = data_config.setdefault("common", {})
    existing_nx = common.get("nx")
    if existing_nx is not None and existing_nx != receptive_field:
        _warn(
            f"Overriding data nx={existing_nx} with model required {receptive_field}"
        )
    common["nx"] = receptive_field
    return data_config


def _iter_split_entries(split_config: _Any):
    if isinstance(split_config, dict):
        yield "train", split_config
        return
    if isinstance(split_config, list):
        for i, item in enumerate(split_config):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected train[{i}] to be a mapping, got {type(item).__name__}"
                )
            yield f"train[{i}]", item
        return
    raise ValueError(
        "Expected data_config['train'] to be a mapping or list of mappings, got "
        f"{type(split_config).__name__}"
    )


def _validate_train_window_lengths(data_config: dict, model_config: dict) -> None:
    net_config = model_config["net"]["config"]
    train_burn_in = net_config.get("train_burn_in")
    train_truncate = net_config.get("train_truncate")
    # ConcatLSTM only consumes train_burn_in inside the truncated-BPTT path; with no
    # train_truncate the full sequence is processed in one (gradient-carrying) pass and
    # burn-in is irrelevant. Nothing to guard against in that case.
    if train_burn_in is None or train_truncate is None:
        return
    # init_dataset merges common into every entry, so ny may live in either place;
    # entry overrides common. ny=None falls back to the full clip length, which is the
    # normal long-window LSTM case and effectively always exceeds burn-in.
    common = data_config.get("common", {})
    common_ny = common.get("ny") if isinstance(common, dict) else None
    for label, train_entry in _iter_split_entries(data_config["train"]):
        ny = train_entry.get("ny", common_ny)
        if ny is None:
            continue
        if not isinstance(ny, int) or isinstance(ny, bool):
            raise ValueError(f"{label}.ny must be an integer, got {ny!r}")
        if ny <= train_burn_in:
            raise ValueError(
                f"{label}.ny (={ny}) must be greater than train_burn_in={train_burn_in}: "
                "otherwise the burn-in window consumes the whole sequence and the member "
                "trains on zero gradient"
            )


def _build_dataloaders(
    data_config: dict,
    learning_config: dict,
    model: _ParametricLightningModule,
    device: _torch.device,
):
    net = model.net
    # getattr (not net.receptive_field) keeps the type checker from widening to nn.Module.
    receptive_field = int(getattr(net, "receptive_field"))
    data_config = _prepare_member_data_config(data_config, receptive_field)
    dataset_train = _init_dataset(data_config, _Split.TRAIN)
    dataset_validation = _init_dataset(data_config, _Split.VALIDATION)

    inner_train = _ConcatDataset(_iter_inner_datasets(dataset_train))
    inner_validation = _ConcatDataset(_iter_inner_datasets(dataset_validation))
    _apply_joint_dataset_hooks(
        dataset_train=inner_train,
        dataset_validation=inner_validation,
        hooks=_get_joint_dataset_hooks(data_config.get("joint", [])),
    )

    setattr(net, "sample_rate", getattr(dataset_train, "sample_rate", None))
    _handshake_datasets(model, dataset_train, dataset_validation)

    batch_plan = _resolve_member_batching(
        learning_config,
        device=device,
        num_train_windows=len(dataset_train),
        num_validation_windows=len(dataset_validation),
    )

    train_loader_config = dict(learning_config["train_dataloader"])
    train_loader_config["capture_grouped_batches"] = False
    train_loader_config["shuffle"] = True
    val_loader_config = dict(learning_config["val_dataloader"])
    val_loader_config["capture_grouped_batches"] = False
    val_loader_config["shuffle"] = False

    return (
        dataset_train,
        dataset_validation,
        _make_parametric_dataloader(dataset_train, train_loader_config),
        _make_parametric_dataloader(dataset_validation, val_loader_config),
        batch_plan,
    )


def _stabilize_checkpoint_path(best_checkpoint: str, member_outdir: _Path) -> _Path:
    if best_checkpoint == "":
        raise RuntimeError(
            f"No best checkpoint was produced for ensemble member output dir {member_outdir}"
        )
    best_checkpoint_path = _Path(best_checkpoint)
    if not best_checkpoint_path.exists():
        raise RuntimeError(f"Best checkpoint does not exist: {best_checkpoint_path}")
    stable_path = member_outdir / "best.ckpt"
    if best_checkpoint_path.resolve() != stable_path.resolve():
        _shutil.copy2(best_checkpoint_path, stable_path)
    return stable_path


def _clear_device_cache(device: _torch.device) -> None:
    if device.type == "cuda":
        _torch.cuda.empty_cache()
        return
    if device.type == "mps":
        mps_module = getattr(_torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "empty_cache"):
            mps_module.empty_cache()


def _param_specs_from_model_config(model_config: dict) -> tuple[_ParamSpec, ...]:
    try:
        raw_specs = model_config["net"]["config"]["params"]
    except KeyError as exc:
        raise ValueError(
            "Model config must define net.config.params for disagreement search"
        ) from exc
    return tuple(_coerce_param_specs(raw_specs))


def _validate_round_idx(round_idx: int) -> None:
    if round_idx < 0:
        raise ValueError(f"round_idx must be non-negative; got {round_idx}")


def _validate_candidate_raw_params(
    candidate: DisagreementCandidate,
    *,
    specs: _Sequence[_ParamSpec],
    switch_idx: tuple[int, ...],
) -> _torch.Tensor:
    if not _math.isfinite(candidate.score):
        raise ValueError(f"Candidate score must be finite; got {candidate.score}")
    if len(candidate.switch_combo) != len(switch_idx):
        raise ValueError(
            f"Candidate switch_combo must have length {len(switch_idx)}; "
            f"got {len(candidate.switch_combo)}"
        )
    raw_params = _torch.as_tensor(candidate.raw_params).detach().cpu().to(_torch.float64)
    if raw_params.ndim != 1:
        raise ValueError(
            f"Candidate raw_params must have shape (P,); got {tuple(raw_params.shape)}"
        )
    if raw_params.shape[0] != len(specs):
        raise ValueError(
            f"Candidate raw_params must have length {len(specs)}; "
            f"got {raw_params.shape[0]}"
        )
    # Clustering groups by switch_combo while decode/quantize read the raw switch columns;
    # cross-check them so an inconsistent candidate can't be grouped as one switch state and
    # emitted as another.
    for position, column in enumerate(switch_idx):
        raw_switch = float(raw_params[column])
        if not raw_switch.is_integer() or int(raw_switch) != candidate.switch_combo[position]:
            raise ValueError(
                f"Candidate raw_params switch column {column} (={raw_switch}) must equal "
                f"switch_combo[{position}]={candidate.switch_combo[position]}"
            )
    return raw_params


def _normalized_continuous_params(
    raw_params: _torch.Tensor,
    *,
    continuous_idx: tuple[int, ...],
    mins: _torch.Tensor,
    widths: _torch.Tensor,
) -> _torch.Tensor:
    if len(continuous_idx) == 0:
        return _torch.empty((0,), dtype=_torch.float64)
    return ((raw_params[list(continuous_idx)] - mins) / widths) * 2.0 - 1.0


def _quantized_dedupe_key(raw_params: _torch.Tensor) -> tuple[float, ...]:
    return tuple(
        round(float(value), _QUANTIZED_DEDUPE_DECIMALS)
        for value in raw_params.detach().cpu().to(_torch.float64).tolist()
    )


def _make_round_y_path(
    round_idx: int,
    params: dict[str, _Any],
    abbreviations: dict[str, str],
    *,
    y_path_prefix: str,
) -> str:
    # Encode the decoded settings into the stem (matching make_starter_settings), keeping
    # the round index in the prefix for provenance: e.g. ``r2_g4.5_bOn.wav``. Selected
    # candidates are quantized to the capture grid and deduped, so the stems are unique
    # within a round; the round prefix disambiguates identical settings across rounds.
    _validate_round_idx(round_idx)
    return _make_capture_y_path(
        f"{y_path_prefix}{round_idx}_", params, abbreviations
    )


def _selected_capture_records(
    selected: _Sequence[DisagreementCandidate],
    specs: tuple[_ParamSpec, ...],
    *,
    round_idx: int,
    y_path_prefix: str,
) -> list[dict[str, _Any]]:
    # An empty selection is a valid no-op round (the acquisition search may exhaust the
    # space): emit_proposals writes an empty list and append_to_data_config copies the
    # config through unchanged, rather than crashing the round.
    _, switch_idx, _ = _split_param_indices(specs)
    abbreviations = _abbreviate_param_names([spec.name for spec in specs])
    records = []
    for candidate in selected:
        raw_params = _validate_candidate_raw_params(
            candidate,
            specs=specs,
            switch_idx=switch_idx,
        )
        params = _decode_named_params(raw_params, specs)
        records.append(
            {
                "params": params,
                "score": float(candidate.score),
                "y_path": _make_round_y_path(
                    round_idx,
                    params,
                    abbreviations,
                    y_path_prefix=y_path_prefix,
                ),
            }
        )
    return records


def _inject_canonical_param_specs(
    data_config: dict,
    model_config: dict,
) -> dict[str, _Any]:
    model_param_specs = _model_param_specs(model_config)
    common = data_config.get("common")
    if not isinstance(common, dict):
        raise ValueError("data_config['common'] must be a mapping")
    common = _deepcopy(common)
    _validate_existing_param_specs_match(common, model_param_specs)
    common["param_specs"] = model_param_specs
    return common


def _make_appended_train_entry(
    template_entry: dict[str, _Any],
    record: dict[str, _Any],
) -> dict[str, _Any]:
    new_entry: dict[str, _Any] = {
        "y_path": record["y_path"],
        "params": _deepcopy(record["params"]),
    }
    for key in _INHERITED_TRAIN_WINDOW_KEYS:
        if key in template_entry:
            new_entry[key] = _deepcopy(template_entry[key])
    return new_entry


def _accepted_capture_plot_path(output_dir: _Path, round_idx: int) -> _Path:
    return output_dir / f"accepted_capture_distributions_round_{round_idx}.png"


def _plot_accepted_capture_distributions(
    *,
    train_entries: _Sequence[dict[str, _Any]],
    specs: _Sequence[_ParamSpec],
    round_idx: int,
    output_dir: _Path,
) -> _Path:
    # Build the figure through the object-oriented API (no pyplot) so plotting never touches
    # matplotlib's global interactive-backend state: a headless active_learn.py run can't
    # hang or fail on GUI-backend selection, and there is no global figure registry to leak.
    figure_module = _importlib.import_module("matplotlib.figure")
    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _accepted_capture_plot_path(output_dir, round_idx)

    fig = figure_module.Figure(figsize=(8.0, 2.75 * len(specs)))
    axes = fig.subplots(len(specs), 1)
    axes_seq = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax, spec in zip(axes_seq, specs):
        values = [entry["params"][spec.name] for entry in train_entries]
        if spec.type == "switch":
            if spec.enum_names is None:
                raise RuntimeError(
                    f"Switch ParamSpec {spec.name!r} is missing enum_names after validation"
                )
            counts = {name: 0 for name in spec.enum_names}
            for value in values:
                if isinstance(value, str):
                    if value not in counts:
                        raise ValueError(
                            f"Train entry params[{spec.name!r}] has unknown enum name {value!r}"
                        )
                    counts[value] += 1
                else:
                    index = int(float(value))
                    if index < 0 or index >= len(spec.enum_names):
                        raise ValueError(
                            f"Train entry params[{spec.name!r}] index {index} is out of range"
                        )
                    counts[spec.enum_names[index]] += 1
            ax.bar(list(counts.keys()), list(counts.values()))
        else:
            numeric_values = [float(value) for value in values]
            if len(set(numeric_values)) == 1:
                center = numeric_values[0]
                half_width = max((spec.max - spec.min) / 20.0, 0.25)
                bins = [center - half_width, center + half_width]
            else:
                bins = min(20, max(5, len(numeric_values)))
            ax.hist(numeric_values, bins=bins)
            ax.set_xlim(min(spec.min, min(numeric_values)), max(spec.max, max(numeric_values)))
        ax.set_title(spec.name)
        ax.set_ylabel("Count")
    axes_seq[-1].set_xlabel("Value")
    fig.tight_layout()
    fig.savefig(output_path)
    return output_path


def _format_capture_checklist(
    entries: _Sequence[dict[str, _Any]],
) -> str:
    lines = ["Capture checklist:", "Train:"]
    for index, entry in enumerate(entries, start=1):
        settings = ", ".join(
            f"{name}={value}" for name, value in entry["params"].items()
        )
        lines.append(f"{index}. {entry['y_path']} -> {settings}")
    return "\n".join(lines)


def _write_json(path: _Path, payload: _Any) -> None:
    with path.open("w") as fp:
        _json.dump(payload, fp, indent=2)
        fp.write("\n")


def _validate_g_opt_args(
    *,
    checkpoint_paths: _Sequence[_Path],
    num_restarts: int,
    num_steps: int,
    g_opt_ny: int,
    g_opt_batch_size: int,
    lr: float,
    z_init_scale: float,
) -> None:
    if len(checkpoint_paths) == 0:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")
    if num_restarts <= 0:
        raise ValueError(f"num_restarts must be positive; got {num_restarts}")
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative; got {num_steps}")
    if g_opt_ny <= 0:
        raise ValueError(f"g_opt_ny must be positive; got {g_opt_ny}")
    if g_opt_batch_size <= 0:
        raise ValueError(
            f"g_opt_batch_size must be positive; got {g_opt_batch_size}"
        )
    if not _math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lr must be a positive finite number; got {lr}")
    if not _math.isfinite(z_init_scale) or z_init_scale <= 0.0:
        raise ValueError(
            f"z_init_scale must be a positive finite number; got {z_init_scale}"
        )


def _load_disagreement_members(
    checkpoint_paths: _Sequence[_Path],
    model_config: dict,
    device: _torch.device,
) -> list[_torch.nn.Module]:
    members: list[_torch.nn.Module] = []
    for checkpoint_path in checkpoint_paths:
        module = _ParametricLightningModule.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
            **_ParametricLightningModule.parse_config(model_config),
        )
        member = module.net.to(device)
        member.requires_grad_(False)
        # Default to eval(); find_disagreement_settings flips members to train() only on the
        # cuda fast path (see _g_opt_cuda_train_mode_safe), where train() is equivalent to
        # eval() but lets cuDNN's RNN backward run. Otherwise eval() + cuDNN disabled is kept
        # so the forward/gradient to z stay identical across devices.
        member.eval()
        members.append(member)
    return members


def _g_opt_cuda_train_mode_safe(
    members: _Sequence[_torch.nn.Module],
    model_config: dict,
) -> bool:
    """Whether the members can run g-opt in train() mode (re-enabling cuDNN) without changing
    results relative to the eval() + cuDNN-disabled path.

    cuDNN's RNN backward refuses to run in eval() mode, so the device-agnostic path disables
    cuDNN on cuda to keep the members in eval() -- correct, but it forces the slow native RNN
    unroll. Running the frozen members in train() instead restores the fast cuDNN kernel, but
    only when train() does not alter the forward/gradient:

      * ``train_truncate is None`` -- ConcatLSTM takes the same full-sequence forward branch
        in train() and eval(); a set ``train_truncate`` would divert it through truncated BPTT
        (detached burn-in + per-chunk-detached hidden state), changing the gradient to z.
      * no dropout / batchnorm -- train() must add no stochasticity or running-stat updates.
    """
    net_config = model_config.get("net", {})
    net_config = net_config.get("config", {}) if isinstance(net_config, dict) else {}
    if not isinstance(net_config, dict) or net_config.get("train_truncate") is not None:
        return False
    for member in members:
        for submodule in member.modules():
            if isinstance(submodule, _torch.nn.RNNBase):
                if float(getattr(submodule, "dropout", 0.0)) > 0.0:
                    return False
            elif isinstance(submodule, _torch.nn.modules.dropout._DropoutNd):
                if float(getattr(submodule, "p", 0.0)) > 0.0:
                    return False
            elif isinstance(submodule, _torch.nn.modules.batchnorm._BatchNorm):
                return False
    return True


def _build_g_opt_batches(
    g_opt_input_wav: str | _Path,
    *,
    g_opt_ny: int,
    g_opt_batch_size: int,
    receptive_field: int,
    device: _torch.device,
) -> tuple[list[_torch.Tensor], int]:
    if g_opt_ny <= receptive_field:
        raise ValueError(
            f"g_opt_ny must be greater than receptive_field={receptive_field}; got {g_opt_ny}"
        )
    signal, wavinfo = _cast(
        tuple[_torch.Tensor, _WavInfo],
        _wav_to_tensor(g_opt_input_wav, info=True),
    )
    if signal.ndim != 1:
        raise ValueError(
            f"Expected mono g-opt input wav to load as shape (L,); got {tuple(signal.shape)}"
        )
    if signal.shape[0] < g_opt_ny:
        raise ValueError(
            f"g-opt input wav must contain at least g_opt_ny={g_opt_ny} samples; "
            f"got {signal.shape[0]}"
        )

    hop = g_opt_ny - receptive_field
    windows = signal.unfold(0, g_opt_ny, hop).contiguous()
    if windows.shape[0] == 0:
        raise RuntimeError("Failed to construct any g-opt windows from the input wav")
    # receptive_field is 1 for the LSTM, so windows are effectively non-overlapping and
    # each is processed from the member's learned initial hidden state (no burn-in warmup
    # at g-opt time, unlike training). The first samples of every window are therefore
    # transient; acceptable for a coarse disagreement proxy.

    loader = _DataLoader(
        _TensorDataset(windows),
        batch_size=g_opt_batch_size,
        shuffle=False,
    )
    batches = [batch[0].to(device) for batch in loader]
    if len(batches) == 0:
        raise RuntimeError("Failed to construct any g-opt batches from the input wav")
    return batches, wavinfo.rate


def _build_mel_transforms(
    *,
    use_mel: bool,
    sample_rate: int,
    device: _torch.device,
):
    if not use_mel:
        return ()

    transforms = []
    for n_fft, hop_length, n_mels in (
        (512, 128, 64),
        (1024, 256, 80),
        (2048, 512, 128),
    ):
        transforms.append(
            _MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            ).to(device)
        )
    return tuple(transforms)


def _disagreement_score(
    outputs: _torch.Tensor,
    mel_transforms,
) -> _torch.Tensor:
    score = outputs.var(dim=0, unbiased=False).mean()
    if len(mel_transforms) == 0:
        return score

    num_members, batch_size, output_len = outputs.shape
    flattened = outputs.reshape(num_members * batch_size, output_len)
    mel_score = _torch.zeros((), device=outputs.device, dtype=outputs.dtype)
    for transform in mel_transforms:
        mel = transform(flattened).reshape(num_members, batch_size, -1)
        mel_score = mel_score + mel.var(dim=0, unbiased=False).mean()
    return score + mel_score / len(mel_transforms)


def _evaluate_disagreement(
    members: _Sequence[_torch.nn.Module],
    x_batch: _torch.Tensor,
    raw_params: _torch.Tensor,
    mel_transforms,
) -> _torch.Tensor:
    outputs = _torch.stack([member(x_batch, raw_params) for member in members], dim=0)
    return _disagreement_score(outputs, mel_transforms)


def _final_disagreement_score(
    members: _Sequence[_torch.nn.Module],
    g_opt_batches: _Sequence[_torch.Tensor],
    raw_params: _torch.Tensor,
    mel_transforms,
) -> float:
    # Rank a candidate on the mean disagreement over the *whole* g-opt signal (no_grad),
    # not the single last training chunk: Task 6's clustering + global top-N selection
    # rides on these scores, so a one-batch estimate would make the ranking noisy.
    with _torch.no_grad():
        total = 0.0
        for batch in g_opt_batches:
            total += float(
                _evaluate_disagreement(members, batch, raw_params, mel_transforms)
                .detach()
                .cpu()
            )
    score = total / len(g_opt_batches)
    if not _math.isfinite(score):
        raise RuntimeError(f"Non-finite disagreement score encountered: {score}")
    return score


def _train_single_member(
    member_idx: int,
    *,
    data_config: dict,
    model_config: dict,
    learning_config: dict,
    outdir: _Path,
    base_seed: int,
    device: _torch.device,
    enable_progress_bar: bool = True,
) -> _Path:
    """Train one ensemble member and return its stabilized ``best.ckpt`` path.

    Module-level (not a closure) so it is picklable for ``spawn``-based parallel
    training. ``data_config`` must already be prepared (``_prepare_data_config``);
    ``learning_config`` is the raw config -- device selection is applied here so each
    worker binds its own assigned ``device`` (e.g. ``cuda:1``).
    """
    _pl.seed_everything(base_seed + member_idx, workers=True)
    learning_config = _prepare_learning_config(learning_config, device)
    member_outdir = _Path(outdir) / f"member_{member_idx:02d}"
    member_outdir.mkdir(parents=True, exist_ok=True)

    dataset_train = None
    dataset_validation = None
    train_dataloader = None
    val_dataloader = None
    trainer = None
    model = None
    try:
        model = _ParametricLightningModule.init_from_config(model_config)
        (
            dataset_train,
            dataset_validation,
            train_dataloader,
            val_dataloader,
            batch_plan,
        ) = _build_dataloaders(data_config, learning_config, model, device)
        batch_plan_path = member_outdir / "batch_plan.json"
        batch_plan_path.write_text(_json.dumps(batch_plan, indent=2) + "\n")
        print(
            f"  member_{member_idx:02d} batch plan on {batch_plan['device_name']}: "
            f"physical={batch_plan['physical_batch_size']}, "
            f"accumulate={batch_plan['accumulate_grad_batches']}, "
            f"effective={batch_plan['effective_batch_size']}",
            flush=True,
        )

        trainer_kwargs = dict(learning_config["trainer"])
        if not enable_progress_bar:
            # Concurrent workers would interleave their live progress bars into an
            # unreadable mess; silence them in the parallel path.
            trainer_kwargs["enable_progress_bar"] = False
        trainer = _pl.Trainer(
            callbacks=_create_parametric_callbacks(learning_config),
            default_root_dir=member_outdir,
            **trainer_kwargs,
        )
        with _filter_warnings("ignore", category=_PossibleUserWarning):
            trainer.fit(
                model,
                train_dataloader,
                val_dataloader,
                **learning_config.get("trainer_fit_kwargs", {}),
            )

        checkpoint_callback = trainer.checkpoint_callback
        best_checkpoint = (
            checkpoint_callback.best_model_path
            if isinstance(checkpoint_callback, _ModelCheckpoint)
            else ""
        )
        return _stabilize_checkpoint_path(best_checkpoint, member_outdir)
    finally:
        if dataset_train is not None:
            dataset_train.teardown()
        if dataset_validation is not None:
            dataset_validation.teardown()
        del val_dataloader
        del train_dataloader
        del trainer
        del model
        _gc.collect()
        _clear_device_cache(device)


def _resolve_ensemble_parallel_plan(
    ensemble_size: int,
    max_workers: int | None,
) -> tuple[int, list[_torch.device]]:
    """Decide how many members train concurrently and which device each one uses.

    Policy (see docs/active_learning_usage.md):
      * ``max_workers`` given -> ``min(max_workers, ensemble_size)`` workers; members
        are round-robined across the available GPUs (a single GPU means every worker
        over-subscribes ``cuda:0`` -- the caller's opt-in, memory is their call).
      * ``max_workers`` is None (default) -> parallel only on multi-GPU CUDA (one
        member per GPU); a single GPU / MPS / CPU stays serial (1 worker) so the
        default never risks OOM by over-subscribing one device.
    """
    if max_workers is not None and max_workers <= 0:
        raise ValueError(f"max_workers must be positive when given; got {max_workers}")

    base_device = _resolve_device()
    gpu_count = _torch.cuda.device_count() if base_device.type == "cuda" else 0

    if max_workers is not None:
        worker_count = min(max_workers, ensemble_size)
    elif base_device.type == "cuda" and gpu_count > 1:
        worker_count = min(ensemble_size, gpu_count)
    else:
        worker_count = 1

    member_devices: list[_torch.device] = []
    for member_idx in range(ensemble_size):
        if base_device.type == "cuda":
            member_devices.append(_torch.device(f"cuda:{member_idx % max(gpu_count, 1)}"))
        else:
            member_devices.append(base_device)
    return worker_count, member_devices


def _coerce_parallel_dataloader_workers(learning_config: dict) -> dict:
    """Force dataloader ``num_workers`` to 0 for parallel training.

    Nested dataloader worker processes spawned from an already-spawned training worker
    are fragile (re-import / start-method hazards); the ensemble members are the unit of
    parallelism here, so collapse any per-member dataloader workers.
    """
    learning_config = _deepcopy(learning_config)
    for key in ("train_dataloader", "val_dataloader"):
        loader_config = learning_config.get(key)
        if not isinstance(loader_config, dict):
            continue
        if loader_config.get("num_workers", 0):
            _warn(
                f"Forcing {key}.num_workers=0 for parallel ensemble training "
                "(nested dataloader workers under spawned member processes are unsafe)."
            )
        loader_config["num_workers"] = 0
        # persistent_workers requires num_workers > 0, so it cannot survive the collapse.
        loader_config.pop("persistent_workers", None)
    return learning_config


def _train_ensemble_parallel(
    *,
    worker_count: int,
    member_devices: _Sequence[_torch.device],
    data_config: dict,
    model_config: dict,
    learning_config: dict,
    outdir: _Path,
    ensemble_size: int,
    base_seed: int,
) -> list[_Path]:
    context = _torch_mp.get_context("spawn")
    results: dict[int, _Path] = {}
    with _ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                _train_single_member,
                member_idx,
                data_config=data_config,
                model_config=model_config,
                learning_config=learning_config,
                outdir=outdir,
                base_seed=base_seed,
                device=member_devices[member_idx],
                enable_progress_bar=False,
            ): member_idx
            for member_idx in range(ensemble_size)
        }
        try:
            for future in _as_completed(futures):
                member_idx = futures[future]
                try:
                    results[member_idx] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Ensemble member {member_idx} failed during parallel training: "
                        f"{exc}"
                    ) from exc
        finally:
            # A failed/interrupted member should not leave siblings running.
            for future in futures:
                future.cancel()
    # Key by member_idx so the returned order (and thus each member's seed) is identical
    # to the serial path regardless of completion order.
    return [results[member_idx] for member_idx in range(ensemble_size)]


def train_ensemble(
    data_config: dict,
    model_config: dict,
    learning_config: dict,
    outdir: _Path,
    *,
    ensemble_size: int = 4,
    base_seed: int = 0,
    max_workers: int | None = None,
) -> list[_Path]:
    """Train a serial (or parallel) ConcatLSTM ensemble; return per-member checkpoints.

    ``max_workers`` controls parallelism (see :func:`_resolve_ensemble_parallel_plan`):
    ``None`` (default) trains serially on one device, or one member per GPU on a
    multi-GPU CUDA box; pass an explicit count to over-subscribe a single GPU (opt-in,
    memory is the caller's responsibility). Results are keyed by member index, so the
    checkpoint order -- and each member's seed -- matches the serial path either way.
    """
    if ensemble_size <= 0:
        raise ValueError(f"ensemble_size must be positive; got {ensemble_size}")
    if model_config["net"]["name"] != "ConcatLSTM":
        raise ValueError(
            "train_ensemble requires model_config['net']['name'] == 'ConcatLSTM'; "
            f"got {model_config['net']['name']!r}"
        )

    outdir = _Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_config = _prepare_data_config(data_config, model_config)
    _validate_train_window_lengths(data_config, model_config)

    worker_count, member_devices = _resolve_ensemble_parallel_plan(
        ensemble_size, max_workers
    )

    if worker_count > 1:
        print(
            f"  Training {ensemble_size} member(s) with {worker_count} parallel "
            f"worker(s) across devices "
            f"{sorted({str(d) for d in member_devices})}.",
            flush=True,
        )
        return _train_ensemble_parallel(
            worker_count=worker_count,
            member_devices=member_devices,
            data_config=data_config,
            model_config=model_config,
            learning_config=_coerce_parallel_dataloader_workers(learning_config),
            outdir=outdir,
            ensemble_size=ensemble_size,
            base_seed=base_seed,
        )

    return [
        _train_single_member(
            member_idx,
            data_config=data_config,
            model_config=model_config,
            learning_config=learning_config,
            outdir=outdir,
            base_seed=base_seed,
            device=member_devices[member_idx],
            enable_progress_bar=True,
        )
        for member_idx in range(ensemble_size)
    ]


def find_disagreement_settings(
    checkpoint_paths: _Sequence[_Path],
    model_config: dict,
    *,
    g_opt_input_wav: str | _Path,
    num_restarts: int = 8,
    num_steps: int = 200,
    g_opt_ny: int = 32768,
    g_opt_batch_size: int = 16,
    lr: float = 0.05,
    z_init_scale: float = 3.0,
    use_mel: bool = False,
    seed: int = 0,
) -> list[DisagreementCandidate]:
    """
    Find high-disagreement control settings with a PANAMA-style query-by-committee search.

    ``z_init_scale`` is the std of the Gaussian latent inits: each restart draws
    ``z ~ N(0, z_init_scale^2)`` and the continuous params are ``min + (max-min)*sigmoid(z)``
    (see ``assemble_raw_params``). A larger scale biases inits toward the saturated
    extremes of sigmoid (often the high-disagreement knob extremes) at the cost of more
    restarts landing in vanishing-gradient regions; the default echoes PANAMA's spread.
    """
    checkpoint_paths = tuple(_Path(path) for path in checkpoint_paths)
    _validate_g_opt_args(
        checkpoint_paths=checkpoint_paths,
        num_restarts=num_restarts,
        num_steps=num_steps,
        g_opt_ny=g_opt_ny,
        g_opt_batch_size=g_opt_batch_size,
        lr=lr,
        z_init_scale=z_init_scale,
    )
    specs = _param_specs_from_model_config(model_config)
    continuous_idx, _, _ = _split_param_indices(specs)

    device = _resolve_device()
    # train_truncate is a training-only BPTT setting. At g-opt time every frozen member runs
    # a single full-sequence forward from its initial state, which is what eval() already does
    # regardless of train_truncate -- so forcing it off does not change any disagreement score.
    # It does let the members run in train() mode (identical forward), which is what re-enables
    # cuDNN's RNN backward on cuda and avoids the slow native per-timestep unroll.
    g_opt_model_config = _deepcopy(model_config)
    _g_opt_net_config = g_opt_model_config.get("net")
    if isinstance(_g_opt_net_config, dict) and isinstance(
        _g_opt_net_config.get("config"), dict
    ):
        _g_opt_net_config["config"]["train_truncate"] = None
    members = _load_disagreement_members(checkpoint_paths, g_opt_model_config, device)
    if len(members) == 0:
        return []

    receptive_fields = {int(getattr(member, "receptive_field")) for member in members}
    if len(receptive_fields) != 1:
        raise RuntimeError(
            f"Expected all ensemble members to share one receptive field; got {receptive_fields}"
        )
    receptive_field = next(iter(receptive_fields))
    g_opt_batches, sample_rate = _build_g_opt_batches(
        g_opt_input_wav,
        g_opt_ny=g_opt_ny,
        g_opt_batch_size=g_opt_batch_size,
        receptive_field=receptive_field,
        device=device,
    )
    mel_transforms = _build_mel_transforms(
        use_mel=use_mel,
        sample_rate=sample_rate,
        device=device,
    )

    generator = _torch.Generator()
    generator.manual_seed(seed)
    n_cont = len(continuous_idx)
    # On cuda, prefer the fast path: flip the (frozen) members to train() so cuDNN's RNN
    # backward runs with cuDNN enabled -- but only when that is provably equivalent to eval()
    # (see _g_opt_cuda_train_mode_safe). Otherwise fall back to eval() + cuDNN disabled, which
    # is correct but runs the slow native RNN unroll. cpu/mps always use the plain eval() path.
    use_cuda_train_mode = device.type == "cuda" and _g_opt_cuda_train_mode_safe(
        members, g_opt_model_config
    )
    if use_cuda_train_mode:
        for member in members:
            member.train()
        print(
            "  g-opt: using cuDNN train() fast path on cuda "
            "(train_truncate=None, no dropout/batchnorm).",
            flush=True,
        )
    elif device.type == "cuda":
        print(
            "  g-opt: using eval() + cuDNN-disabled path on cuda (slow native RNN unroll); "
            "set train_truncate=None and drop dropout/batchnorm to enable the fast path.",
            flush=True,
        )
    cudnn_ctx = (
        _torch.backends.cudnn.flags(enabled=False)
        if device.type == "cuda" and not use_cuda_train_mode
        else _contextlib.nullcontext()
    )
    candidates: list[DisagreementCandidate] = []
    switch_combos = list(_switch_combinations(specs))
    num_combos = len(switch_combos)
    try:
        with cudnn_ctx:
            for combo_idx, switch_combo in enumerate(switch_combos, start=1):
                print(
                    f"  g-opt: switch combo {combo_idx}/{num_combos} "
                    f"{switch_combo}",
                    flush=True,
                )
                if n_cont == 0:
                    # No continuous latents to ascend: the setting is fully determined by
                    # the switch combo, so every restart/step would reproduce the identical
                    # candidate. Evaluate once and move on.
                    final_raw_params = _assemble_raw_params(
                        _torch.zeros((0,), dtype=_torch.float32, device=device),
                        switch_combo,
                        specs,
                    )
                    candidates.append(
                        DisagreementCandidate(
                            raw_params=final_raw_params.detach().cpu(),
                            switch_combo=switch_combo,
                            score=_final_disagreement_score(
                                members, g_opt_batches, final_raw_params, mel_transforms
                            ),
                        )
                    )
                    continue
                for restart_idx in range(num_restarts):
                    z = (
                        z_init_scale
                        * _torch.randn(
                            (n_cont,),
                            generator=generator,
                            dtype=_torch.float32,
                        )
                    ).to(device)
                    z.requires_grad_(True)
                    optimizer = _torch.optim.Adam([z], lr=lr)
                    progress = _tqdm(
                        range(num_steps),
                        desc=f"  combo {combo_idx}/{num_combos} restart "
                        f"{restart_idx + 1}/{num_restarts}",
                        leave=False,
                    )
                    for step in progress:
                        batch = g_opt_batches[step % len(g_opt_batches)]
                        raw_params = _assemble_raw_params(z, switch_combo, specs)
                        score = _evaluate_disagreement(
                            members, batch, raw_params, mel_transforms
                        )
                        optimizer.zero_grad(set_to_none=True)
                        (-score).backward()
                        optimizer.step()
                        progress.set_postfix(disagreement=f"{float(score):.4e}")

                    final_raw_params = _assemble_raw_params(z, switch_combo, specs)
                    candidates.append(
                        DisagreementCandidate(
                            raw_params=final_raw_params.detach().cpu(),
                            switch_combo=switch_combo,
                            score=_final_disagreement_score(
                                members, g_opt_batches, final_raw_params, mel_transforms
                            ),
                        )
                    )
    finally:
        del mel_transforms
        del g_opt_batches
        del members
        _gc.collect()
        _clear_device_cache(device)

    return candidates


def cluster_and_select(
    candidates: _Sequence[DisagreementCandidate],
    model_config: dict,
    *,
    max_per_round: int,
    cluster_threshold: float = _DEFAULT_CLUSTER_THRESHOLD,
) -> list[DisagreementCandidate]:
    """
    Greedily cluster high-disagreement candidates within each switch combination.

    ``cluster_threshold`` is the maximum L2 distance in normalized continuous
    ``[-1, 1]`` space for two candidates to be treated as one cluster; the default
    ``0.1`` keeps nearby knob settings together while still allowing distinct proposals.

    The clustering space is a fixed per-param ``[min, max] -> [-1, 1]`` linear map, chosen
    deliberately so the threshold weights every continuous knob equally and stays
    independent of each spec's model-side ``normalized_min/max`` tuning. It is *not* the
    model's ``_encode_params`` space and is not required to match it.
    """
    if max_per_round <= 0:
        raise ValueError(f"max_per_round must be positive; got {max_per_round}")
    if not _math.isfinite(cluster_threshold) or cluster_threshold < 0.0:
        raise ValueError(
            "cluster_threshold must be a non-negative finite number; "
            f"got {cluster_threshold}"
        )

    specs = _param_specs_from_model_config(model_config)
    continuous_idx, switch_idx, _ = _split_param_indices(specs)
    if len(candidates) == 0:
        return []

    grouped_candidates: dict[tuple[int, ...], list[DisagreementCandidate]] = {}
    for candidate in candidates:
        grouped_candidates.setdefault(candidate.switch_combo, []).append(candidate)

    mins = _torch.tensor(
        [specs[index].min for index in continuous_idx],
        dtype=_torch.float64,
    )
    widths = _torch.tensor(
        [specs[index].max - specs[index].min for index in continuous_idx],
        dtype=_torch.float64,
    )
    if len(continuous_idx) > 0 and _torch.any(widths <= 0.0):
        raise ValueError("Continuous ParamSpecs must satisfy max > min for clustering")

    representatives: list[DisagreementCandidate] = []
    for grouped in grouped_candidates.values():
        sorted_group = sorted(grouped, key=lambda candidate: candidate.score, reverse=True)
        if len(continuous_idx) == 0:
            representatives.append(
                DisagreementCandidate(
                    raw_params=_validate_candidate_raw_params(
                        sorted_group[0],
                        specs=specs,
                        switch_idx=switch_idx,
                    ).to(_torch.float32),
                    switch_combo=sorted_group[0].switch_combo,
                    score=sorted_group[0].score,
                )
            )
            continue

        kept_norms: list[_torch.Tensor] = []
        # Greedy threshold-grouping adapted from PANAMA's cluster_gs /
        # group_similar_vectors: walk candidates best-first and keep the first
        # representative whose normalized continuous setting is farther than the threshold.
        for candidate in sorted_group:
            raw_params = _validate_candidate_raw_params(
                candidate,
                specs=specs,
                switch_idx=switch_idx,
            )
            normalized = _normalized_continuous_params(
                raw_params,
                continuous_idx=continuous_idx,
                mins=mins,
                widths=widths,
            )
            if any(
                _torch.linalg.vector_norm(normalized - kept).item() <= cluster_threshold
                for kept in kept_norms
            ):
                continue
            kept_norms.append(normalized)
            representatives.append(
                DisagreementCandidate(
                    raw_params=raw_params.to(_torch.float32),
                    switch_combo=candidate.switch_combo,
                    score=candidate.score,
                )
            )

    deduped: dict[tuple[float, ...], DisagreementCandidate] = {}
    for candidate in representatives:
        quantized = _quantize_to_capture_grid(candidate.raw_params, specs).to(_torch.float32)
        quantized_candidate = DisagreementCandidate(
            raw_params=quantized.detach().cpu(),
            switch_combo=candidate.switch_combo,
            score=candidate.score,
        )
        dedupe_key = _quantized_dedupe_key(quantized_candidate.raw_params)
        existing = deduped.get(dedupe_key)
        if existing is None or quantized_candidate.score > existing.score:
            deduped[dedupe_key] = quantized_candidate

    return sorted(
        deduped.values(),
        key=lambda candidate: candidate.score,
        reverse=True,
    )[:max_per_round]


def emit_proposals(
    selected: _Sequence[DisagreementCandidate],
    model_config: dict,
    *,
    round_idx: int,
    output_dir: _Path,
    y_path_prefix: str = "r",
) -> tuple[_Path, list[dict[str, _Any]]]:
    """
    Write the selected settings for one active-learning round as a human-facing proposal
    list and print a capture checklist.

    Returns ``(proposals_path, proposals)``. The returned ``proposals`` are the canonical
    per-capture records (decoded named params + suggested ``y_path``); feed them straight
    into :func:`append_to_data_config` so the y_paths the human is told to capture are the
    *same* ones written into the aggregated config (rather than independently re-deriving
    them and risking silent divergence).
    """
    _validate_round_idx(round_idx)
    specs = _param_specs_from_model_config(model_config)
    records = _selected_capture_records(
        selected,
        specs,
        round_idx=round_idx,
        y_path_prefix=y_path_prefix,
    )
    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"proposed_captures_round_{round_idx}.json"
    _write_json(output_path, records)
    print(_format_capture_checklist(records))
    return output_path, records


def append_to_data_config(
    prev_data_config: dict,
    proposals: _Sequence[dict[str, _Any]],
    model_config: dict,
    *,
    round_idx: int,
    output_dir: _Path,
    plot: bool = True,
) -> tuple[dict, _Path]:
    """
    Append one training entry per emitted proposal to a parametric ``data.json`` config,
    preserving ``common`` and ``validation`` while canonicalizing ``train`` to a list.

    ``proposals`` are the records returned by :func:`emit_proposals`. Consuming them here
    (instead of re-deriving from the candidates) makes the proposals file and the aggregated
    config share one source of truth, so a capture's suggested ``y_path`` cannot differ
    between what the human is told to record and what training expects.

    Returns ``(new_data_config, aggregated_config_path)``. When ``plot`` is true an accepted-
    capture distribution PNG is also written next to the config (PANAMA parity); its path is
    deterministic (``_accepted_capture_plot_path``). Plotting is a non-load-bearing side
    effect: a plotting failure is warned about, never raised, so it can't abort the round.

    Note: ``common.nx`` is not injected here; the training pipeline
    (``_prepare_member_data_config``) sets it from the model's receptive field at train time.
    A standalone ``init_dataset`` on this config therefore needs ``common.nx`` supplied.
    """
    _validate_round_idx(round_idx)
    if prev_data_config.get("type") != "parametric":
        raise ValueError(
            "prev_data_config must be a parametric data config "
            "(type='parametric')"
        )
    specs = _param_specs_from_model_config(model_config)

    new_data_config = _deepcopy(prev_data_config)
    new_data_config["common"] = _inject_canonical_param_specs(new_data_config, model_config)

    if "train" not in new_data_config:
        raise ValueError("prev_data_config must define a train split")
    train_entries = [
        _deepcopy(entry) for _, entry in _iter_split_entries(new_data_config["train"])
    ]
    if len(train_entries) == 0:
        raise ValueError("prev_data_config['train'] must contain at least one entry")
    template_entry = train_entries[0]
    train_entries.extend(
        _make_appended_train_entry(template_entry, record) for record in proposals
    )
    new_data_config["train"] = train_entries

    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"aggregated_data_config_{round_idx}.json"
    _write_json(output_path, new_data_config)
    if plot:
        try:
            _plot_accepted_capture_distributions(
                train_entries=train_entries,
                specs=specs,
                round_idx=round_idx,
                output_dir=output_dir,
            )
        except Exception as exc:  # pragma: no cover - plotting is cosmetic parity only
            _warn(f"Failed to plot accepted-capture distributions: {exc}")
    return new_data_config, output_path


@_dataclass(frozen=True)
class RoundResult:
    """Outcome of one active-learning round (see :func:`run_round`)."""

    round_idx: int
    checkpoint_paths: list[_Path]
    candidates: list[DisagreementCandidate]
    selected: list[DisagreementCandidate]
    proposals: list[dict[str, _Any]]
    proposals_path: _Path
    aggregated_data_config: dict[str, _Any]
    aggregated_config_path: _Path


def _resolve_g_opt_input_wav(
    g_opt_input_wav: str | _Path | None,
    data_config: dict,
) -> str | _Path:
    # PANAMA reamps one fixed input clip at every proposed setting; default it to the
    # config's own input (common.x_path) so the g-opt search runs over the same signal the
    # captures are recorded against (plan Open Question 2 default).
    if g_opt_input_wav is not None:
        return g_opt_input_wav
    common = data_config.get("common", {})
    x_path = common.get("x_path") if isinstance(common, dict) else None
    if not x_path:
        raise ValueError(
            "g_opt_input_wav was not provided and data_config['common']['x_path'] is "
            "missing; pass an explicit g-opt input wav or set common.x_path"
        )
    return x_path


def run_round(
    *,
    round_idx: int,
    output_dir: str | _Path,
    data_config: dict,
    model_config: dict,
    learning_config: dict,
    g_opt_input_wav: str | _Path | None = None,
    ensemble_size: int = 4,
    num_restarts: int = 8,
    num_steps: int = 200,
    max_per_round: int = 5,
    g_opt_ny: int = 32768,
    g_opt_batch_size: int = 16,
    g_opt_lr: float = 0.05,
    use_mel: bool = False,
    cluster_threshold: float = _DEFAULT_CLUSTER_THRESHOLD,
    seed: int = 0,
    checkpoint_paths: _Sequence[str | _Path] | None = None,
    y_path_prefix: str = "r",
    plot: bool = True,
    max_workers: int | None = None,
) -> RoundResult:
    """
    Run one active-learning round end to end (PANAMA-style query-by-committee).

    Sequences the Task 4-6 building blocks: (1) train (or reuse) a serial ConcatLSTM
    ensemble, (2) search for high-disagreement control settings, (3) cluster/quantize/select
    the top proposals, and (4) emit a human-facing proposal list plus an aggregated
    ``data.json`` for the next round. One call == one round; the human then records the
    proposed captures, fills the placeholder ``y_path``s, and reruns with ``round_idx + 1``
    against the returned ``aggregated_data_config``.

    ``seed`` drives both the ensemble member seeds (``base_seed``) and the g-opt latent
    inits, so a round is reproducible from a single value. ``g_opt_input_wav`` defaults to
    ``data_config['common']['x_path']``. Pass ``checkpoint_paths`` to skip retraining and
    reuse an already-trained ensemble (e.g. resuming a round after a g-opt crash).

    Quantization to the capture grid happens inside :func:`cluster_and_select`, never here or
    in the g-opt loop (plan D5).
    """
    _validate_round_idx(round_idx)
    net_name = model_config.get("net", {}).get("name") if isinstance(
        model_config.get("net"), dict
    ) else None
    if net_name != "ConcatLSTM":
        raise ValueError(
            "run_round requires model_config['net']['name'] == 'ConcatLSTM'; "
            f"got {net_name!r}"
        )

    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Re-running a round regenerates placeholder y_paths; if the human has already filled
    # them in the aggregated config, that hand-entered data is about to be overwritten.
    aggregated_config_target = output_dir / f"aggregated_data_config_{round_idx}.json"
    if aggregated_config_target.exists():
        _warn(
            f"{aggregated_config_target} already exists and will be overwritten; any "
            "y_paths you filled in for this round will be regenerated as placeholders. "
            "Back it up first if you have edited it."
        )

    # Snapshot the resolved inputs for provenance before anything mutates them.
    for basename, config in (
        ("data", data_config),
        ("model", model_config),
        ("learning", learning_config),
    ):
        _write_json(output_dir / f"round_{round_idx}_input_{basename}_config.json", config)

    resolved_g_opt_wav = _resolve_g_opt_input_wav(g_opt_input_wav, data_config)

    # Record the round's own knobs (not captured by the config snapshots) so a round is
    # reproducible from this file alone. Written before training so it survives a crash.
    _write_json(
        output_dir / f"round_{round_idx}_run_args.json",
        {
            "round_idx": round_idx,
            "g_opt_input_wav": str(resolved_g_opt_wav),
            "ensemble_size": ensemble_size,
            "num_restarts": num_restarts,
            "num_steps": num_steps,
            "max_per_round": max_per_round,
            "g_opt_ny": g_opt_ny,
            "g_opt_batch_size": g_opt_batch_size,
            "g_opt_lr": g_opt_lr,
            "use_mel": use_mel,
            "cluster_threshold": cluster_threshold,
            "seed": seed,
            "y_path_prefix": y_path_prefix,
            "max_workers": max_workers,
            "reused_checkpoint_paths": (
                None if checkpoint_paths is None else [str(p) for p in checkpoint_paths]
            ),
        },
    )

    if checkpoint_paths is None:
        print(
            f"[round {round_idx}] Training ensemble of {ensemble_size} ConcatLSTM "
            "member(s)...",
            flush=True,
        )
        resolved_checkpoints = train_ensemble(
            data_config,
            model_config,
            learning_config,
            output_dir / f"ensemble_round_{round_idx}",
            ensemble_size=ensemble_size,
            base_seed=seed,
            max_workers=max_workers,
        )
        print(
            f"[round {round_idx}] Ensemble training complete "
            f"({len(resolved_checkpoints)} checkpoint(s)).",
            flush=True,
        )
    else:
        resolved_checkpoints = [_Path(path) for path in checkpoint_paths]
        if len(resolved_checkpoints) == 0:
            raise ValueError("checkpoint_paths, when provided, must be non-empty")
        print(
            f"[round {round_idx}] Reusing {len(resolved_checkpoints)} provided "
            "checkpoint(s); skipping training.",
            flush=True,
        )

    print(
        f"[round {round_idx}] Starting g-optimization (disagreement search): "
        f"{num_restarts} restart(s) x {num_steps} step(s) per switch combo.",
        flush=True,
    )
    candidates = find_disagreement_settings(
        resolved_checkpoints,
        model_config,
        g_opt_input_wav=resolved_g_opt_wav,
        num_restarts=num_restarts,
        num_steps=num_steps,
        g_opt_ny=g_opt_ny,
        g_opt_batch_size=g_opt_batch_size,
        lr=g_opt_lr,
        use_mel=use_mel,
        seed=seed,
    )

    print(
        f"[round {round_idx}] g-optimization complete: {len(candidates)} candidate(s). "
        "Clustering and selecting proposals...",
        flush=True,
    )
    selected = cluster_and_select(
        candidates,
        model_config,
        max_per_round=max_per_round,
        cluster_threshold=cluster_threshold,
    )
    if len(selected) < max_per_round:
        # cluster_and_select only ever shrinks the pool (it clusters/dedupes the
        # num_restarts x num_combos candidates and caps at max_per_round); it can never
        # invent more. So a short round means the *candidate generation* came up short --
        # the lever is --num-restarts (more raw candidates per switch combo), not
        # --max-per-round (just the cap).
        _warn(
            f"Selected {len(selected)} proposal(s) but max_per_round={max_per_round} was "
            f"requested. The {len(candidates)} candidate(s) from {num_restarts} restart(s) "
            "collapsed to fewer distinct settings after clustering/dedup. Increase "
            "--num-restarts (or --cluster-threshold) to propose more captures this round."
        )

    proposals_path, proposals = emit_proposals(
        selected,
        model_config,
        round_idx=round_idx,
        output_dir=output_dir,
        y_path_prefix=y_path_prefix,
    )

    aggregated_data_config, aggregated_config_path = append_to_data_config(
        data_config,
        proposals,
        model_config,
        round_idx=round_idx,
        output_dir=output_dir,
        plot=plot,
    )

    return RoundResult(
        round_idx=round_idx,
        checkpoint_paths=resolved_checkpoints,
        candidates=candidates,
        selected=selected,
        proposals=proposals,
        proposals_path=proposals_path,
        aggregated_data_config=aggregated_data_config,
        aggregated_config_path=aggregated_config_path,
    )
