import json as _json
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from time import time as _time
from collections.abc import Sequence as _Sequence
from typing import Optional as _Optional
from warnings import warn as _warn

import librosa as _librosa
import matplotlib.pyplot as _plt
import pytorch_lightning as _pl
import torch as _torch
from lightning_fabric.utilities.warnings import PossibleUserWarning as _PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Sampler as _Sampler

from nam.data import ConcatDataset as _ConcatDataset
from nam.data import Dataset as _Dataset
from nam.data import Split as _Split
from nam.data import apply_joint_dataset_hooks as _apply_joint_dataset_hooks
from nam.data import get_joint_dataset_hooks as _get_joint_dataset_hooks
from nam.data import init_dataset as _init_dataset
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric import ParametricDataset as _ParametricDataset
from nam.models.parametric import ParametricNet as _ParametricNet
from nam.models.parametric import bake as _bake
from nam.models.parametric import data_config_from_model as _data_config_from_model
from nam.models.parametric import export_parametric as _export_parametric
from nam.models.parametric import output_scale_from_datasets as _output_scale_from_datasets
from nam.train.core import _ValidationStopping
from nam.train.full import _create_callbacks
from nam.train.full import _handshake_datasets
from nam.train.full import _rms as _rms
from nam.train.lightning_module import LightningModule as _LightningModule
from nam.train.lightning_module import LossConfig as _LossConfig
from nam.train.lightning_module import _LossItem
from nam.util import filter_warnings as _filter_warnings

# NB: the global RNG seed is set once by `nam.train.full` (imported above); no need to
# reseed here.


def _iter_inner_datasets(dataset) -> tuple[_Dataset, ...]:
    if isinstance(dataset, _Dataset):
        return (dataset,)
    if isinstance(dataset, _ParametricDataset):
        return _iter_inner_datasets(dataset.dataset)
    if isinstance(dataset, _ConcatDataset):
        inner = []
        for child in dataset.datasets:
            inner.extend(_iter_inner_datasets(child))
        return tuple(inner)
    raise TypeError(
        "Expected a Dataset, ParametricDataset, or ConcatDataset; "
        f"got {type(dataset).__name__}"
    )


# Default-on. Each capture pairs one control setting with many audio windows, and the train
# ConcatDataset lays the captures out in contiguous index ranges. HyperWaveNet._run_conditioned
# groups a batch by unique control setting and runs one functional_call per group, so a batch
# that mixes captures fans out into several tiny GPU passes. Keeping every batch inside one
# capture collapses that to a single full-batch functional_call per step.
_CAPTURE_GROUPED_KEY = "capture_grouped_batches"
# Metrics are tracked per data bucket: the training split is audio and control
# settings the model fit on; the validation split is held out on both axes.
_TRAIN_BUCKET = "seen_audio_seen_params"
_VALIDATION_BUCKET = "unseen_audio_unseen_params"
# Means reduced as sample-weighted averages across the epoch. ESR is handled
# separately because it is a ratio of energies, not a mean.
_BUCKET_MEAN_METRICS = ("MSE", "MRSTFT")


class _MelSpectrogram(_torch.nn.Module):
    """Pure-torch mel spectrogram equivalent to ``torchaudio.transforms.MelSpectrogram``
    with its default HTK, unnormalized filterbank and centered reflection-padded STFT.

    The PANAMA mel objectives previously depended on torchaudio, whose prebuilt wheel loads
    a compiled extension that must match the installed torch ABI exactly; a mismatched pair
    fails to ``dlopen`` at runtime. This reimplements the only piece those objectives use
    (an STFT magnitude raised to ``power`` and projected onto a mel filterbank) using
    ``torch.stft`` plus a librosa-built filterbank, so training needs no torchaudio.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        power: float = 1.0,
    ):
        super().__init__()
        self._n_fft = int(n_fft)
        self._win_length = int(win_length)
        self._hop_length = int(hop_length)
        self._power = float(power)
        self.register_buffer("_window", _torch.hann_window(self._win_length))
        # (n_mels, n_freqs); htk + norm=None matches torchaudio's melscale_fbanks defaults.
        filterbank = _librosa.filters.mel(
            sr=sample_rate, n_fft=self._n_fft, n_mels=n_mels, htk=True, norm=None
        )
        self.register_buffer(
            "_mel_fb", _torch.tensor(filterbank, dtype=_torch.float32)
        )

    def forward(self, waveform: _torch.Tensor) -> _torch.Tensor:
        original = waveform.shape
        spec = _torch.stft(
            waveform.reshape(-1, original[-1]),
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=self._window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        spectrogram = spec.abs().pow(self._power)  # (B, n_freqs, n_frames)
        mel = _torch.matmul(self._mel_fb, spectrogram)  # (B, n_mels, n_frames)
        return mel.reshape(*original[:-1], mel.shape[-2], mel.shape[-1])


class _CaptureBatchSampler(_Sampler):
    """
    Yield batches whose sample indices all fall inside one capture's contiguous range, so a
    HyperWaveNet step sees a single control setting and runs one functional_call over the
    whole batch instead of one per unique setting.

    Trade-off vs. global shuffling: rows are shuffled *within* a capture and the batch order
    is shuffled across captures, but a batch never mixes captures. Uses the global torch RNG
    (seeded once by ``nam.train.full``) so the order is reproducible yet varies per epoch.

    Since every batch is capture-pure, each ``__iter__`` call also records which capture
    every yielded batch belongs to (``last_batch_captures``), so a caller holding a reference
    to this sampler can recover a batch's capture from its ``batch_idx`` -- including across
    epochs, where ``shuffle=True`` reorders both the rows and the batch list on every call.
    This is how ``_EpochMetrics`` groups batches by capture without re-deriving the mapping
    itself (see ``_ParametricLightningModule._capture_index``).
    """

    def __init__(
        self,
        capture_lengths: _Sequence[int],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        self._capture_lengths = tuple(int(length) for length in capture_lengths)
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        # Populated by __iter__; index i is the capture index of the i-th batch of the most
        # recently started iteration (epoch).
        self.last_batch_captures: tuple[int, ...] = ()

    def _capture_offsets(self) -> tuple[int, ...]:
        offsets = []
        offset = 0
        for length in self._capture_lengths:
            offsets.append(offset)
            offset += length
        return tuple(offsets)

    def __iter__(self):
        batches: list[list[int]] = []
        batch_captures: list[int] = []
        for capture_idx, (offset, length) in enumerate(
            zip(self._capture_offsets(), self._capture_lengths)
        ):
            if length == 0:
                continue
            if self._shuffle:
                order = [offset + i for i in _torch.randperm(length).tolist()]
            else:
                order = list(range(offset, offset + length))
            for start in range(0, length, self._batch_size):
                batch = order[start : start + self._batch_size]
                if self._drop_last and len(batch) < self._batch_size:
                    continue
                batches.append(batch)
                batch_captures.append(capture_idx)
        if self._shuffle:
            permutation = _torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in permutation]
            batch_captures = [batch_captures[i] for i in permutation]
        # Set before yielding the first batch: everything above runs eagerly on the first
        # `next()` call, since it all precedes the `yield from` below.
        self.last_batch_captures = tuple(batch_captures)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for length in self._capture_lengths:
            if self._drop_last:
                total += length // self._batch_size
            else:
                total += (length + self._batch_size - 1) // self._batch_size
        return total

    def capture_for_batch(self, batch_idx: int) -> int:
        """Which capture the ``batch_idx``-th batch of the current iteration came from."""
        return self.last_batch_captures[batch_idx]


def _capture_lengths(dataset) -> tuple[int, ...]:
    if isinstance(dataset, _ConcatDataset):
        return tuple(len(child) for child in dataset.datasets)
    return (len(dataset),)


def _make_parametric_dataloader(dataset, loader_config: dict) -> _DataLoader:
    loader_config = dict(loader_config)
    capture_grouped = loader_config.pop(_CAPTURE_GROUPED_KEY, True)
    if not capture_grouped:
        return _DataLoader(dataset, **loader_config)

    # batch_sampler is mutually exclusive with these DataLoader args, so the sampler owns them.
    batch_size = loader_config.pop("batch_size", 1)
    shuffle = loader_config.pop("shuffle", False)
    drop_last = loader_config.pop("drop_last", False)
    loader_config.pop("sampler", None)
    batch_sampler = _CaptureBatchSampler(
        _capture_lengths(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return _DataLoader(dataset, batch_sampler=batch_sampler, **loader_config)


def _get_parametric_net(model: _LightningModule) -> _ParametricNet:
    net = model.net
    if not isinstance(net, _ParametricNet):
        raise TypeError(
            "Parametric training expects model_config['net']['name'] to initialize "
            f"a ParametricNet, got {type(net).__name__}"
        )
    return net


class _EnergyAccumulator:
    """Pool metrics over a set of batches by summing energies, exactly as
    ``_EpochMetrics`` used to do over an entire split.

    ESR is summed as energies -- the squared error and squared target are summed
    separately and divided only at the end -- so a near-silent batch cannot divide
    by ~zero and poison the pool the way a per-batch ESR average does. MSE and
    MRSTFT/Mel are reduced as sample-weighted means over the same batches. This is
    the correct reduction *within* one capture (every window shares one control
    setting, so there is no cross-capture level mismatch to worry about); see
    ``_EpochMetrics`` for why pooling energies *across* captures is a defect.
    """

    def __init__(self) -> None:
        self._err_sq: _Optional[_torch.Tensor] = None
        self._tgt_sq: _Optional[_torch.Tensor] = None
        self._weighted: dict[str, _torch.Tensor] = {}
        self.rows = 0

    def update(
        self,
        preds: _torch.Tensor,
        targets: _torch.Tensor,
        means: dict[str, _torch.Tensor],
    ) -> None:
        preds = preds.detach()
        targets = targets.detach()
        err_sq = _torch.sum(_torch.square(preds - targets))
        tgt_sq = _torch.sum(_torch.square(targets))
        self._err_sq = err_sq if self._err_sq is None else self._err_sq + err_sq
        self._tgt_sq = tgt_sq if self._tgt_sq is None else self._tgt_sq + tgt_sq
        rows = targets.shape[0]
        self.rows += rows
        for key, value in means.items():
            # Equal-length windows make a row-count weighting exact for per-element
            # means like MSE and consistent for MRSTFT/Mel.
            contrib = value.detach() * rows
            self._weighted[key] = (
                contrib if key not in self._weighted else self._weighted[key] + contrib
            )

    def compute(self) -> dict[str, float]:
        if self.rows == 0:
            return {}
        tgt_sq = float(self._tgt_sq)
        out: dict[str, float] = {
            "ESR": float(self._err_sq) / tgt_sq if tgt_sq > 0.0 else float("inf")
        }
        for key, total in self._weighted.items():
            out[key] = float(total) / self.rows
        return out


class _EpochMetrics:
    """Accumulate per-batch metrics across an epoch and reduce them once at the end,
    using a two-level reduction across captures.

    A parametric split lays out several captures (one control setting each) back to
    back, and their output levels can differ by many dB. Pooling energies (or taking a
    sample-weighted mean) directly across ALL windows of ALL captures -- what this class
    used to do -- lets the loud captures swamp the denominator, so a quiet capture's own
    ESR barely moves the reported number even though it can be terrible. The fix keeps
    the original pooling *within* a capture (see ``_EnergyAccumulator``) -- that part was
    never wrong, a near-silent capture still can't divide by ~zero -- and then takes an
    equal-weight mean of the per-capture results, so every capture counts the same
    regardless of how loud it is.

    This requires knowing which capture each batch belongs to. Callers that can supply a
    ``capture_index`` to ``update`` get the two-level treatment. Callers that cannot (no
    ``_CaptureBatchSampler`` behind the dataloader -- see
    ``_ParametricLightningModule._capture_index``) fall back to the single-level pooling
    over the whole epoch, matching the old behavior exactly rather than silently mixing
    the two reductions.
    """

    def __init__(self) -> None:
        self._captures: dict[int, _EnergyAccumulator] = {}
        # Fallback accumulator, used only when capture identity is unavailable.
        self._ungrouped = _EnergyAccumulator()
        self._capture_grouped: _Optional[bool] = None

    def update(
        self,
        preds: _torch.Tensor,
        targets: _torch.Tensor,
        means: dict[str, _torch.Tensor],
        capture_index: _Optional[int] = None,
    ) -> None:
        grouped = capture_index is not None
        if self._capture_grouped is None:
            self._capture_grouped = grouped
        elif self._capture_grouped != grouped:
            raise RuntimeError(
                "_EpochMetrics received a capture_index on some updates but not others "
                "within the same epoch; capture identity must be available for either "
                "every batch or none of them"
            )
        if grouped:
            accumulator = self._captures.setdefault(capture_index, _EnergyAccumulator())
        else:
            accumulator = self._ungrouped
        accumulator.update(preds, targets, means)

    def compute(self) -> dict[str, float]:
        if not self._capture_grouped:
            return self._ungrouped.compute()
        per_capture = [
            accumulator.compute()
            for accumulator in self._captures.values()
            if accumulator.rows > 0
        ]
        if not per_capture:
            return {}
        keys: dict[str, None] = {}
        for metrics in per_capture:
            keys.update(dict.fromkeys(metrics))
        out: dict[str, float] = {}
        for key in keys:
            values = [metrics[key] for metrics in per_capture if key in metrics]
            out[key] = sum(values) / len(values)
        return out


def _bucket_means(loss_dict: dict) -> dict[str, _torch.Tensor]:
    return {
        key: loss_dict[key].value
        for key in _BUCKET_MEAN_METRICS
        if key in loss_dict and loss_dict[key].value is not None
    }


@_dataclass
class _ParametricLossConfig(_LossConfig):
    mel_weight: _Optional[float] = None

    @classmethod
    def parse_config(cls, config):
        parsed = super().parse_config(config)
        parsed["mel_weight"] = config.get("mel_weight")
        return parsed


class _ParametricLightningModule(_LightningModule):
    def __init__(
        self,
        net,
        optimizer_config=None,
        scheduler_config=None,
        loss_config=None,
    ):
        super().__init__(
            net,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            loss_config=(
                _ParametricLossConfig() if loss_config is None else loss_config
            ),
        )
        self._mel_mrstft = None
        # Set via `set_capture_batch_samplers` once the dataloaders exist (see `main`).
        # `None` means "capture identity unavailable" -- `_EpochMetrics` then falls back
        # to single-level pooling over the whole split instead of a wrong two-level split.
        self._train_capture_sampler: _Optional[_CaptureBatchSampler] = None
        self._val_capture_sampler: _Optional[_CaptureBatchSampler] = None

    @classmethod
    def parse_config(cls, config):
        parsed = super().parse_config(config)
        parsed["loss_config"] = _ParametricLossConfig.init_from_config(
            config.get("loss", {})
        )
        return parsed

    def set_capture_batch_samplers(
        self,
        train_sampler: _Optional[_Sampler] = None,
        val_sampler: _Optional[_Sampler] = None,
    ) -> None:
        """Give the module a reference to the (possibly capture-grouped) batch
        samplers backing its dataloaders, so `_EpochMetrics` can pool ESR/MSE/MRSTFT
        per capture rather than over the whole split. Only a `_CaptureBatchSampler`
        (`capture_grouped_batches` not disabled) actually enables the two-level
        reduction; anything else is treated the same as passing `None`.
        """
        self._train_capture_sampler = train_sampler
        self._val_capture_sampler = val_sampler

    @staticmethod
    def _capture_index(
        sampler: _Optional[_Sampler], batch_idx: int
    ) -> _Optional[int]:
        if isinstance(sampler, _CaptureBatchSampler):
            return sampler.capture_for_batch(batch_idx)
        return None

    def _get_loss_dict(self, preds, targets):
        loss_dict = super()._get_loss_dict(preds, targets)
        if self._loss_config.mel_weight is not None:
            loss_dict["Mel"] = _LossItem(
                self._loss_config.mel_weight,
                self._mel_mrstft_loss(preds, targets),
            )
        return loss_dict

    def _mel_mrstft_loss(
        self, preds: _torch.Tensor, targets: _torch.Tensor
    ) -> _torch.Tensor:
        """PANAMA's seven-resolution log-mel L1 loss."""
        if self._mel_mrstft is None:
            sample_rate = self.net.sample_rate
            if sample_rate is None:
                raise RuntimeError("mel_weight requires the model sample rate")
            self._mel_mrstft = [
                _MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    win_length=window_length,
                    n_mels=n_mels,
                    power=1.0,
                ).to(preds.device)
                for window_length, n_mels in zip(
                    (32, 64, 128, 256, 512, 1024, 2048),
                    (5, 10, 20, 40, 80, 160, 320),
                )
            ]
        return sum(
            _torch.nn.functional.l1_loss(
                _torch.log10(transform(preds) + 1e-5),
                _torch.log10(transform(targets) + 1e-5),
            )
            for transform in self._mel_mrstft
        )

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        # A non-finite gradient poisons Adam's moment estimates. Marking every gradient None
        # excludes the parameters from this step entirely: weights, weight decay, moments,
        # and per-parameter step counters all stay unchanged.
        non_finite = any(
            p.grad is not None and not _torch.isfinite(p.grad).all()
            for p in self.parameters()
        )
        if non_finite:
            for parameter in self.parameters():
                parameter.grad = None
        super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )

    def configure_optimizers(self):
        # Mirrors the base wiring but (a) uses AdamW for decoupled weight decay -- the
        # optimizer_config's "weight_decay" curbs the proxy's train/val overfitting, and
        # weight_decay 0.0 (the default) is identical to plain Adam -- and (b) adds optional
        # linear LR warmup via scheduler_config["warmup_epochs"], which keeps the earliest
        # updates small so full-BPTT LSTM training settles onto the descent before full-size
        # steps, avoiding the early gradient blow-ups that otherwise drive a run to NaN.
        optimizer_config = {"weight_decay": 0.0, **self._optimizer_config}
        optimizer = _torch.optim.AdamW(self.parameters(), **optimizer_config)
        if self._scheduler_config is None:
            return optimizer
        base_scheduler = getattr(
            _torch.optim.lr_scheduler, self._scheduler_config["class"]
        )(optimizer, **self._scheduler_config["kwargs"])
        warmup_epochs = self._scheduler_config.get("warmup_epochs") or 0
        if warmup_epochs > 0:
            warmup = _torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / warmup_epochs,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = _torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, base_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = base_scheduler
        lr_scheduler_config = {"scheduler": scheduler}
        for key in ("interval", "frequency", "monitor"):
            if key in self._scheduler_config:
                lr_scheduler_config[key] = self._scheduler_config[key]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_train_epoch_start(self):
        self._train_metrics = _EpochMetrics()

    def training_step(self, batch, batch_idx):
        preds, targets, loss_dict = self._shared_step(batch)
        loss = _torch.zeros((), device=preds.device)
        for v in loss_dict.values():
            if v.weight is not None and v.weight > 0.0:
                if v.value is None:
                    raise RuntimeError("Weighted training losses must define a tensor value")
                loss = loss + v.weight * v.value
        capture_index = self._capture_index(self._train_capture_sampler, batch_idx)
        self._train_metrics.update(preds, targets, _bucket_means(loss_dict), capture_index)
        return loss

    def on_train_epoch_end(self):
        self._log_bucket(self._train_metrics.compute(), _TRAIN_BUCKET)

    def on_validation_epoch_start(self):
        self._val_metrics = _EpochMetrics()

    def validation_step(self, batch, batch_idx):
        preds, targets, loss_dict = self._shared_step(batch)
        capture_index = self._capture_index(self._val_capture_sampler, batch_idx)
        self._val_metrics.update(preds, targets, _bucket_means(loss_dict), capture_index)
        # Reduction happens once in on_validation_epoch_end; nothing to return.

    def on_validation_epoch_end(self):
        metrics = self._val_metrics.compute()
        if not metrics:
            return
        self._log_bucket(metrics, _VALIDATION_BUCKET)
        val_loss_key = self._val_loss_key()
        if val_loss_key not in metrics:
            raise RuntimeError(
                f"Validation loss {val_loss_key!r} was not accumulated this epoch"
            )
        # Bare keys back the checkpoint monitor (val_loss), the checkpoint filename
        # (ESR, MSE), and any threshold-ESR early stopping.
        monitored = dict(metrics)
        monitored["val_loss"] = metrics[val_loss_key]
        self.log_dict(monitored)

    def _val_loss_key(self) -> str:
        val_loss_type = self._loss_config.val_loss
        if isinstance(val_loss_type, str):
            return val_loss_type.upper()
        return val_loss_type.value.upper()

    def _log_bucket(self, metrics: dict[str, float], bucket: str) -> None:
        if not metrics:
            return
        self.log_dict({f"{name}/{bucket}": value for name, value in metrics.items()})


def _create_parametric_callbacks(learning_config):
    callbacks = _create_callbacks(learning_config, packed=False)
    threshold_esr = learning_config.get("threshold_esr")
    if threshold_esr is not None:
        callbacks.append(
            _ValidationStopping(monitor="ESR", stopping_threshold=threshold_esr)
        )
    return callbacks


def _parametric_plot_label(ds: _ParametricDataset) -> str:
    y_path = getattr(ds.dataset, "_y_path", None)
    if y_path is not None:
        return _Path(y_path).name
    params = ", ".join(f"{value:.3g}" for value in ds.params.detach().cpu().tolist())
    return f"params=[{params}]"


def _plot_parametric(
    model: _LightningModule,
    ds,
    savefig=None,
    show=True,
    window_start: _Optional[int] = None,
    window_end: _Optional[int] = None,
):
    if isinstance(ds, _ConcatDataset):

        def extend_savefig(i, original):
            if original is None:
                return None
            original = _Path(original)
            return original.with_name(
                f"{original.stem}_{i}{original.suffix}"
            )

        for i, ds_i in enumerate(ds.datasets):
            _plot_parametric(
                model,
                ds_i,
                savefig=extend_savefig(i, savefig),
                show=show and i == len(ds.datasets) - 1,
                window_start=window_start,
                window_end=window_end,
            )
        return
    if not isinstance(ds, _ParametricDataset):
        raise TypeError(
            "Expected a ParametricDataset or ConcatDataset of ParametricDataset; "
            f"got {type(ds).__name__}"
        )
    with _torch.no_grad():
        sample_rate = ds.sample_rate if ds.sample_rate is not None else 48_000
        tx = len(ds.dataset.x) / sample_rate
        print(f"Run (t={tx:.2f})")
        t0 = _time()
        output = model(ds.dataset.x, ds.params).cpu().numpy().flatten()
        t1 = _time()
        try:
            rt = f"{tx / (t1 - t0):.2f}"
        except ZeroDivisionError:
            rt = "???"
        print(f"Took {t1 - t0:.2f} ({rt}x)")
    target = ds.dataset.y.cpu().numpy()
    # Held-out validation clips can be short (a tail of unseen audio). If the requested
    # window starts past the end of this clip the default slice plots nothing, so fall back
    # to the whole clip rather than an empty figure. ESR below is over the full signal.
    if window_start is not None and window_start >= len(output):
        window_start, window_end = None, None
    _plt.figure(figsize=(16, 5))
    _plt.plot(output[window_start:window_end], label="Prediction")
    _plt.plot(target[window_start:window_end], linestyle="--", label="Target")
    nrmse = _rms(_torch.tensor(output) - ds.dataset.y) / _rms(ds.dataset.y)
    esr = nrmse**2
    _plt.title(f"{_parametric_plot_label(ds)}\nESR={esr:.3f}")
    _plt.legend()
    if savefig is not None:
        _plt.savefig(savefig)
    if show:
        _plt.show()


def main(
    data_config,
    model_config,
    learning_config,
    outdir: _Path,
    no_show: bool = False,
    make_plots=True,
):
    if not outdir.exists():
        raise RuntimeError(f"No output location found at {outdir}")
    for basename, config in (
        ("data", data_config),
        ("model", model_config),
        ("learning", learning_config),
    ):
        with open(_Path(outdir, f"config_{basename}.json"), "w") as fp:
            _json.dump(config, fp, indent=4)

    if model_config["net"]["name"] == "PackedWaveNet":
        raise ValueError("PackedWaveNet is not supported by the parametric trainer")

    data_config = _data_config_from_model(data_config, model_config)
    model = _ParametricLightningModule.init_from_config(model_config)
    net = _get_parametric_net(model)

    data_config["common"] = data_config.get("common", {})
    if "nx" in data_config["common"]:
        _warn(
            f"Overriding data nx={data_config['common']['nx']} with model required "
            f"{net.receptive_field}"
        )
    data_config["common"]["nx"] = net.receptive_field

    dataset_train = _init_dataset(data_config, _Split.TRAIN)
    dataset_validation = _init_dataset(data_config, _Split.VALIDATION)
    inner_train = _ConcatDataset(_iter_inner_datasets(dataset_train))
    inner_validation = _ConcatDataset(_iter_inner_datasets(dataset_validation))
    _apply_joint_dataset_hooks(
        dataset_train=inner_train,
        dataset_validation=inner_validation,
        hooks=_get_joint_dataset_hooks(data_config.get("joint", [])),
    )
    net.sample_rate = getattr(dataset_train, "sample_rate", None)
    _handshake_datasets(model, dataset_train, dataset_validation)

    train_dataloader = _make_parametric_dataloader(
        dataset_train, learning_config["train_dataloader"]
    )
    val_dataloader = _make_parametric_dataloader(
        dataset_validation, learning_config["val_dataloader"]
    )
    # Lets _EpochMetrics pool ESR/MSE/MRSTFT per capture instead of over the whole split;
    # a no-op (single-level fallback) if capture_grouped_batches was turned off for a loader.
    model.set_capture_batch_samplers(
        train_sampler=train_dataloader.batch_sampler,
        val_sampler=val_dataloader.batch_sampler,
    )
    callbacks = _create_parametric_callbacks(learning_config)
    trainer = _pl.Trainer(
        callbacks=callbacks,
        default_root_dir=outdir,
        **learning_config["trainer"],
    )

    try:
        try:
            with _filter_warnings("ignore", category=_PossibleUserWarning):
                trainer.fit(
                    model,
                    train_dataloader,
                    val_dataloader,
                    **learning_config.get("trainer_fit_kwargs", {}),
                )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        # Reached on normal completion or a user interrupt; in both cases we still want to
        # export the best model. A hard training error instead propagates past this block
        # (skipping export so a secondary bake/export failure can't mask the real error)
        # while the outer `finally` still tears the datasets down.
        checkpoint_callback = trainer.checkpoint_callback
        best_checkpoint = (
            checkpoint_callback.best_model_path
            if isinstance(checkpoint_callback, _ModelCheckpoint)
            else ""
        )
        if best_checkpoint != "":
            model = _ParametricLightningModule.load_from_checkpoint(
                best_checkpoint,
                **_ParametricLightningModule.parse_config(model_config),
            )
        net = _get_parametric_net(model)
        model.cpu()
        model.eval()
        net.sample_rate = getattr(dataset_train, "sample_rate", None)
        _handshake_datasets(model, dataset_train, dataset_validation)
        if make_plots:
            _plot_parametric(
                model,
                dataset_validation,
                savefig=_Path(outdir, "comparison.png"),
                window_start=100_000,
                window_end=110_000,
                show=False,
            )
            _plot_parametric(model, dataset_validation, show=not no_show)

        output_scale = _output_scale_from_datasets((dataset_train, dataset_validation))
        # Baking a fixed-setting stock WaveNet snapshot is a HyperWaveNet-only feature;
        # other parametric nets (e.g. ConcatWaveNet) only get the parametric export.
        if isinstance(net, _HyperWaveNet):
            _bake(
                net,
                net.nominal_params,
                output_scale=output_scale,
            ).export(outdir)
        _export_parametric(
            net,
            outdir,
            basename="model_parametric",
            output_scale=output_scale,
        )
    finally:
        dataset_train.teardown()
        dataset_validation.teardown()
