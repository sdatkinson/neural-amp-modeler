"""
Parametric WaveNet wrapper built from a frozen template plus generated weight deltas.

Exported weight layout:
1. Inner WaveNet weights in the stock WaveNet export order.
2. Hypernetwork serialized state: parameters in ``named_parameters()`` order, then the
   low-rank anchor buffer.
"""

from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from contextlib import nullcontext as _nullcontext
from typing import Any as _Any
from typing import Optional as _Optional
from typing import cast as _cast

import numpy as _np
import torch as _torch
from torch.func import functional_call as _functional_call

from .._from_nam import convert_nam_wavenet_config as _convert_nam_wavenet_config
from ..wavenet._wavenet import WaveNet as _WaveNet
from ._base import ParametricNet as _ParametricNet
from ._hypernet import Hypernetwork as _Hypernetwork
from ._spec import ParamSpec as _ParamSpec

_DEFAULT_HYPERNET_SELECTOR = {"exclude_suffixes": ["_conv.weight"]}
_DELTA_MAP_KEY = "delta_map"
_WeightsLike = _Sequence[float] | _np.ndarray | _torch.Tensor


class HyperWaveNet(_ParametricNet):
    def __init__(
        self,
        *,
        template: _WaveNet,
        hypernet: _Hypernetwork,
        param_specs: _Sequence[_ParamSpec],
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(param_specs=param_specs, sample_rate=sample_rate)
        self._validate_supported_template(template)
        self._template = template
        self._hypernet = hypernet
        # `named_parameters()` walks the whole module tree; the weight dict is rebuilt on every
        # step, so hold the mapping instead. Safe to cache: `.to()`, `load_state_dict()` and
        # `import_weights()` all write through the same Parameter objects.
        self._template_parameters = dict(template.named_parameters())
        self._uniform_batch_params = False
        self._uniform_batch_params_checked = False

    @classmethod
    def parse_config(cls, config: dict[str, _Any]) -> dict[str, _Any]:
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        raw_params = config.pop("params", None)
        if raw_params is None:
            raise ValueError("HyperWaveNet config must define a params array")
        param_specs = tuple(_ParamSpec.from_dict(spec) for spec in raw_params)
        if len(param_specs) == 0:
            raise ValueError("HyperWaveNet config must define at least one ParamSpec")
        if "condition_dsp" in config:
            # Block this until HyperWaveNet has an end-to-end nested-DSP story. WaveNet
            # exports condition_dsp weights inside config["condition_dsp"]["weights"],
            # but HyperWaveNet rebuilds its template from config only via
            # convert_nam_wavenet_config()->WaveNet.init_from_config(), which would
            # silently drop those nested weights and leave the condition DSP random.
            raise NotImplementedError(
                "HyperWaveNet does not support condition_dsp because nested WaveNet "
                "condition_dsp weights are not copied into the parametric template"
            )

        hypernet_config = dict(config.pop("hypernet", {}))
        hypernet_config.pop(_DELTA_MAP_KEY, None)
        if "selector" not in hypernet_config:
            hypernet_config["selector"] = dict(_DEFAULT_HYPERNET_SELECTOR)

        wavenet_config = _convert_nam_wavenet_config(config, sample_rate=sample_rate)
        template = _WaveNet.init_from_config(wavenet_config)
        hypernet = _Hypernetwork.from_config(
            input_dim=sum(spec.num_inputs for spec in param_specs),
            named_shapes={
                name: parameter.shape
                for name, parameter in template.named_parameters()
            },
            config=hypernet_config,
        )
        return {
            "template": template,
            "hypernet": hypernet,
            "param_specs": param_specs,
            "sample_rate": sample_rate,
        }

    @staticmethod
    def _validate_supported_template(template: _WaveNet) -> None:
        if template._condition_dsp is not None:
            raise NotImplementedError(
                "HyperWaveNet does not support WaveNet templates with condition_dsp"
            )

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._template.receptive_field

    def _forward_mps_safe(self, x: _torch.Tensor, **kwargs) -> _torch.Tensor:
        # Slimmable templates draw ONE random channel width per forward (stock
        # `context_adjust_to_random` semantics). Set it here -- ABOVE BaseNet's MPS >65,536
        # stitching loop -- so every temporal segment of a single logical forward shares one
        # width, matching single-width inference. (Drawing it in `_run_conditioned` instead
        # would let each MPS segment redraw a width.) Only meaningful while training; eval
        # pins the full width, so non-slimmable models hit `_nullcontext` with no overhead.
        context = (
            self._template.context_adjust_to_random()
            if self.training and self._template.is_slimmable()
            else _nullcontext()
        )
        with context:
            return super()._forward_mps_safe(x, **kwargs)

    def set_uniform_batch_params(self, uniform: bool) -> None:
        """
        Promise that a batched ``params`` tensor holds one setting repeated down the batch.

        That is what ``_CaptureBatchSampler`` already guarantees during parametric training
        (every batch is drawn from a single capture). Without the promise, ``_run_conditioned``
        has to *discover* the grouping from the data -- ``torch.unique`` on a device tensor,
        then boolean masking -- and each of those reads a GPU-computed value back to the host.
        Those syncs stop the CPU from queuing work ahead of the GPU, which is exactly what
        hides per-op launch latency on a model this small. Given the promise we can take
        ``p[0]`` and run the batch under one weight set with no host round-trip at all.

        The promise is verified against the first batched call, so a wrong one fails loudly
        rather than silently training every row at the wrong setting.
        """
        self._uniform_batch_params = bool(uniform)
        self._uniform_batch_params_checked = False

    @property
    def supports_compiled_step(self) -> bool:
        # Worth more here than anywhere else: the generated deltas are ~100 tensors of a few
        # dozen elements each, so the step is dominated by launching kernels too small to pay
        # for their own launch. See `ParametricNet.set_compiled`.
        return True

    def _check_uniform_batch_params(self, p: _torch.Tensor) -> None:
        if self._uniform_batch_params_checked:
            return
        # One host sync, on the first batched call only.
        self._uniform_batch_params_checked = True
        if not bool(_torch.equal(p, p[:1].expand_as(p))):
            raise RuntimeError(
                "set_uniform_batch_params(True) promises every row of a batched params "
                "tensor is the same control setting, but this batch mixed settings. Turn "
                "off capture-grouped batching's uniform fast path (or fix the sampler) "
                "before training."
            )

    def _compilable_step(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        """One setting (1-D ``p``) applied to the whole of ``x``."""
        return self._apply_conditioned_weights(x, self._conditioned_weight_dict(p))

    def _run_conditioned(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        if p.ndim == 1:
            return self._run_step(x, p)

        if x.shape[0] != p.shape[0]:
            raise ValueError(
                f"Input batch size {x.shape[0]} must match encoded params batch size {p.shape[0]}"
            )

        if self._uniform_batch_params:
            self._check_uniform_batch_params(p)
            return self._run_step(x, p[0])

        # Batched parametric training often repeats the same control setting across many
        # windows from one capture. Group those rows so each unique setting runs one
        # functional_call over a real mini-batch instead of one call per sample.
        unique_p, inverse = _torch.unique(p, dim=0, return_inverse=True)
        deltas = self._hypernet.generate(unique_p)
        outputs = None
        for i in range(unique_p.shape[0]):
            mask = inverse == i
            y_group = self._apply_conditioned_weights(
                x[mask].contiguous(),
                self._assemble_weight_dict(
                    {name: delta[i] for name, delta in deltas.items()}
                ),
            )
            if outputs is None:
                outputs = y_group.new_empty((x.shape[0], y_group.shape[1]))
            outputs[mask] = y_group
        if outputs is None:
            raise RuntimeError("Expected at least one unique parameter row")
        return outputs

    def _assemble_weight_dict(
        self, deltas: _Mapping[str, _torch.Tensor]
    ) -> dict[str, _torch.Tensor]:
        weight_dict = dict(self._template_parameters)
        for name, delta in deltas.items():
            weight_dict[name] = weight_dict[name] + delta
        return weight_dict

    def _conditioned_weight_dict(self, p: _torch.Tensor) -> dict[str, _torch.Tensor]:
        """
        ``_assemble_weight_dict(hypernet.generate(p))`` for a single (1-D) setting, done a
        shape-group at a time: one stack + one add per group rather than one add per target.
        """
        if p.ndim != 1:
            raise ValueError(
                f"Expected a single encoded setting of shape (E,); got {tuple(p.shape)}"
            )
        parameters = self._template_parameters
        weight_dict = dict(parameters)
        for group, stacked in self._hypernet.generate_groups(p):
            names = group.names
            if len(names) == 1:
                weight_dict[names[0]] = parameters[names[0]] + stacked.squeeze(
                    group.stack_dim
                )
                continue
            conditioned = _torch.stack([parameters[name] for name in names]) + stacked
            for name, weight in zip(names, conditioned.unbind(group.stack_dim)):
                weight_dict[name] = weight
        return weight_dict

    def _apply_conditioned_weights(
        self,
        x: _torch.Tensor,
        weight_dict: dict[str, _torch.Tensor],
    ) -> _torch.Tensor:
        y = _functional_call(self._template, weight_dict, (x[:, None, :],))
        if y.shape[1] != 1:
            raise RuntimeError(
                f"Expected template WaveNet to return one channel; got shape {tuple(y.shape)}"
            )
        return y[:, 0, :]

    def _export_inner_config(self) -> dict[str, _Any]:
        config = self._template.export_config(sample_rate=self.sample_rate)
        hypernet_config = dict(self._hypernet.config)
        hypernet_config[_DELTA_MAP_KEY] = self._export_delta_map()
        config["hypernet"] = hypernet_config
        return config

    def _export_delta_map(self) -> list[dict[str, _Any]]:
        export_offsets = {}
        offset = 0
        for name, parameter in self._template.named_parameters():
            export_offsets[name] = offset
            offset += parameter.numel()

        delta_map = []
        for target in self._hypernet.export_target_metadata():
            name = _cast(str, target["name"])
            if name not in export_offsets:
                raise RuntimeError(
                    f"Hypernetwork target {name!r} is missing from the template export order"
                )
            entry = dict(target)
            entry["export_offset"] = export_offsets[name]
            delta_map.append(entry)
        return delta_map

    def _export_weights(self) -> _np.ndarray:
        tensors = [
            _torch.from_numpy(self._template.export_weights()).to(dtype=_torch.float32)
        ]
        tensors.append(self._hypernet.export_state())
        return _torch.cat(tensors).numpy()

    def import_weights(self, weights: _WeightsLike, i: int = 0) -> int:
        weights_tensor = _cast(
            _torch.Tensor,
            weights if isinstance(weights, _torch.Tensor) else _torch.tensor(weights),
        )
        if weights_tensor.ndim != 1:
            raise ValueError(
                f"HyperWaveNet weights must be a flat 1-D sequence; got shape {tuple(weights_tensor.shape)}"
            )

        i = self._template.import_weights(weights_tensor, i)
        # Tail-ownership assumption: the hypernet weights are expected to run to the END of
        # the blob, so `remaining` is everything after the base. `remaining == 0` seeds the
        # base only (e.g. from a stock WaveNet checkpoint) and leaves the hypernet as-is;
        # `remaining == expected` is a full parametric restore. Unlike the stock
        # `import_weights(weights, i) -> next_i` contract this rejects a blob with trailing
        # weights owned by another module, so HyperWaveNet can't yet be embedded as a
        # sub-component of a larger (e.g. packed) blob.
        remaining = len(weights_tensor) - i
        expected = self._hypernet.state_count()
        if remaining == 0:
            return i
        if remaining != expected:
            raise ValueError(
                f"Expected either 0 or {expected} hypernetwork weights after the base "
                f"WaveNet blob, but found {remaining}"
            )
        return self._hypernet.import_state(weights_tensor, i)
