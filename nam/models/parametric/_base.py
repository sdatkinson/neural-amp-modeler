"""
Generic parametric model wrapper utilities.
"""

import abc as _abc
import math as _math
from collections.abc import Sequence as _Sequence
from typing import Any as _Any
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import cast as _cast
from warnings import warn as _warn

import numpy as _np
import torch as _torch
import torch.nn.functional as _F

from .._abc import ImportsWeights as _ImportsWeights
from ..base import BaseNet as _BaseNet
from ._spec import ParamSpec as _ParamSpec

# Devices with no inductor backend. Compilation stays enabled and simply falls back, so one
# learning config can be shared between a Mac and a CUDA training box.
_UNCOMPILABLE_DEVICES = frozenset({"mps"})


class ParametricNet(_BaseNet, _ImportsWeights):
    def __init__(
        self,
        param_specs: _Sequence[_ParamSpec],
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(sample_rate=sample_rate)
        if len(param_specs) == 0:
            raise ValueError("param_specs must contain at least one parameter definition")
        param_names = tuple(spec.name for spec in param_specs)
        if len(set(param_names)) != len(param_names):
            raise ValueError("param_specs must define unique parameter names")
        self._param_specs = tuple(param_specs)
        self._param_names = param_names
        self._encoded_param_dim = sum(spec.num_inputs for spec in self._param_specs)
        # Underscore-prefixed to match the repo's buffer convention (e.g. BaseNet's
        # `_sample_rate`). These names become state_dict keys, so keep them private and
        # stable; expose read access via the `nominal_params` property below.
        self.register_buffer(
            "_param_mins",
            _torch.tensor([spec.min for spec in self._param_specs], dtype=_torch.float32),
        )
        self.register_buffer(
            "_param_maxs",
            _torch.tensor([spec.max for spec in self._param_specs], dtype=_torch.float32),
        )
        self.register_buffer(
            "_nominal_params",
            _torch.tensor(
                [spec.default for spec in self._param_specs], dtype=_torch.float32
            ),
        )
        self._compiled_step = None
        self._warned_uncompilable_device = False

    @property
    def supports_compiled_step(self) -> bool:
        """Whether ``set_compiled`` has an inner forward it can hand to ``torch.compile``.

        Subclasses opt in by overriding this together with ``_compilable_step``. Nets whose
        conditioned forward carries Python control flow that dynamo would only graph-break
        on (``ConcatLSTM``'s truncated-BPTT block loop, say) should leave it False.
        """
        return False

    def set_compiled(self, enabled: bool, **compile_kwargs: _Any) -> None:
        """
        Route this net's inner conditioned forward through ``torch.compile``.

        These models issue many small kernels -- small enough that launching them costs the
        CPU more than running them costs the GPU -- so fusing the step (and with
        ``mode="reduce-overhead"``, replaying it as a CUDA graph) is worth more than the
        arithmetic it saves.

        Only the inner forward is wrapped, never the module, so export, checkpointing and
        ``state_dict`` keys are untouched and disabling this restores the eager path exactly.
        On by default in the example parametric learning config; the trainer applies it for
        the duration of `fit` from the ``torch_compile`` block in learning.json.

        Enabling this is always safe: a device inductor cannot serve (MPS) falls back to the
        eager step at call time, so the same config runs on a Mac and on a CUDA box.
        """
        if not enabled:
            self._compiled_step = None
            return
        if not self.supports_compiled_step:
            _warn(
                f"{type(self).__name__} does not support a compiled step; "
                "ignoring the torch_compile setting."
            )
            self._compiled_step = None
            return
        self._compiled_step = _torch.compile(self._compilable_step, **compile_kwargs)

    def _compilable_step(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        """The tensor work a compile-capable subclass wants fused."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define a compilable step"
        )

    def _run_step(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        step = self._compiled_step
        # Inductor fails to lower the WaveNet's dilated convs when grad is off
        # ("LoweringException: NotImplementedError: View"), which is how Lightning runs
        # validation. Training steps are the hot path and compile cleanly, so take the
        # compiled route only there and leave inference on the eager one.
        if step is None or not _torch.is_grad_enabled():
            return self._compilable_step(x, p)
        if x.device.type in _UNCOMPILABLE_DEVICES:
            self._warn_uncompilable_device(x.device.type)
            return self._compilable_step(x, p)
        return step(x, p)

    def _warn_uncompilable_device(self, device_type: str) -> None:
        # Checked per call rather than in `set_compiled` because the trainer enables
        # compilation before Lightning moves the model to its accelerator -- at that point
        # every parameter is still on CPU and the real device is not yet knowable.
        if self._warned_uncompilable_device:
            return
        self._warned_uncompilable_device = True
        _warn(
            f"torch.compile has no inductor backend for {device_type}; "
            f"{type(self).__name__} is running the eager step instead."
        )

    @property
    def param_specs(self) -> tuple[_ParamSpec, ...]:
        return self._param_specs

    @property
    def param_names(self) -> tuple[str, ...]:
        return self._param_names

    @property
    def param_dim(self) -> int:
        return len(self._param_specs)

    @property
    def encoded_param_dim(self) -> int:
        return self._encoded_param_dim

    @property
    def nominal_params(self) -> _torch.Tensor:
        """Raw (un-encoded) default parameter values, in declared order."""
        return _cast(_torch.Tensor, self._nominal_params)

    @property
    @_abc.abstractmethod
    def pad_start_default(self) -> bool:
        pass

    @property
    @_abc.abstractmethod
    def receptive_field(self) -> int:
        pass

    def _encode_params(self, raw: _torch.Tensor) -> _torch.Tensor:
        param_mins = _cast(_torch.Tensor, self._param_mins)
        param_maxs = _cast(_torch.Tensor, self._param_maxs)
        raw = _torch.as_tensor(
            raw, dtype=param_mins.dtype, device=param_mins.device
        )
        if raw.ndim not in (1, 2):
            raise ValueError(
                f"Expected raw params to have shape (P,) or (B, P); got {tuple(raw.shape)}"
            )
        if raw.shape[-1] != self.param_dim:
            raise ValueError(
                f"Expected raw params trailing dimension {self.param_dim}; got {raw.shape[-1]}"
            )

        encoded = []
        for i, spec in enumerate(self._param_specs):
            if spec.type == "switch":
                switch_values = raw[..., i]
                if not _torch.isfinite(switch_values).all():
                    raise ValueError(
                        f"Switch parameter {spec.name!r} index must be finite"
                    )
                rounded = _torch.round(switch_values)
                if (switch_values != rounded).any():
                    raise ValueError(
                        f"Switch parameter {spec.name!r} index must be an integer"
                    )
                indices = rounded.to(dtype=_torch.long)
                if ((indices < 0) | (indices >= spec.num_inputs)).any():
                    raise ValueError(
                        f"Switch parameter {spec.name!r} index must be within "
                        f"[0, {spec.num_inputs - 1}]"
                    )
                encoded.append(
                    _F.one_hot(indices, num_classes=spec.num_inputs).to(dtype=param_mins.dtype)
                )
                continue

            value = raw[..., i : i + 1]
            param_min = param_mins[i]
            param_max = param_maxs[i]
            frac = (value - param_min) / (param_max - param_min)
            encoded.append(
                spec.normalized_min
                + frac * (spec.normalized_max - spec.normalized_min)
            )

        return _torch.cat(encoded, dim=-1)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: _torch.Tensor,
        params: _torch.Tensor,
        pad_start: _Optional[bool] = None,
    ) -> _torch.Tensor:
        """
        Run the model at the given control settings.

        Supported shapes (B = batch, L = samples, P = param_dim, L' = L - receptive_field + 1):

        - x (L,),   params (P,)   -> y (L',)     one clip at one setting.
        - x (B, L), params (B, P) -> y (B, L')   per-sample settings; batches must align 1:1.
        - x (L,),   params (B, P) -> y (B, L')   one clip evaluated at B settings (broadcast).
        - x (B, L), params (P,)   -> y (B, L')   one shared setting applied to B clips.

        A batched ``params`` (B, P) paired with a (B', L) input where B != B' is an error:
        batched settings must align 1:1 with batched audio. The (P,) form is the only way one
        setting spans multiple clips. This contract is what lets a subclass that generates
        per-sample weights rely on strict alignment instead of implicit broadcasting.
        """
        pad_start = self.pad_start_default if pad_start is None else pad_start
        scalar_input = x.ndim == 1
        params = _torch.as_tensor(params)
        if params.ndim not in (1, 2):
            raise ValueError(
                f"Expected params to have shape (P,) or (B, P); got {tuple(params.shape)}"
            )
        if scalar_input:
            x = x[None]
            if params.ndim == 2:
                x = x.expand(params.shape[0], -1)
        elif params.ndim == 2 and x.shape[0] != params.shape[0]:
            raise ValueError(
                f"Input batch size {x.shape[0]} must match params batch size {params.shape[0]}"
            )
        if pad_start:
            x = _torch.cat(
                (_torch.zeros((len(x), self.receptive_field - 1)).to(x.device), x),
                dim=1,
            )
        if x.shape[1] < self.receptive_field:
            raise ValueError(
                f"Input has {x.shape[1]} samples, which is too few for this model with "
                f"receptive field {self.receptive_field}!"
            )
        y = self._forward_mps_safe(x, params=params)
        if scalar_input and params.ndim == 1:
            y = y[0]
        return y

    def _forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: _torch.Tensor, *, params: _torch.Tensor
    ) -> _torch.Tensor:
        return self._run_conditioned(x, self._encode_params(params))

    @_abc.abstractmethod
    def _run_conditioned(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        pass

    def _at_nominal_settings(self, x: _torch.Tensor) -> _torch.Tensor:
        return self(x, self.nominal_params)

    def _export_config(self) -> dict[str, _Any]:
        config = self._export_inner_config()
        config["params"] = [spec.to_dict() for spec in self._param_specs]
        return config

    @_abc.abstractmethod
    def _export_inner_config(self) -> dict[str, _Any]:
        pass

    def _export_input_output(self) -> _Tuple[_np.ndarray, _np.ndarray]:
        rate = self.sample_rate
        if rate is None:
            raise RuntimeError(
                "Cannot export model's input and output without a sample rate."
            )
        num_samples = int(rate)
        device = _cast(_torch.Tensor, self._param_mins).device
        x = _torch.cat(
            [
                _torch.zeros((num_samples,), device=device),
                0.5
                * _torch.sin(
                    2.0
                    * _math.pi
                    * 220.0
                    * _torch.linspace(0.0, 1.0, num_samples + 1, device=device)[:-1]
                ),
                _torch.zeros((num_samples,), device=device),
            ]
        )
        with _torch.no_grad():
            training = self.training
            self.eval()
            y = self(x, self.nominal_params, pad_start=True)
            self.train(training)
        return x.detach().cpu().numpy(), y.detach().cpu().numpy()

    @_abc.abstractmethod
    def _export_weights(self) -> _np.ndarray:
        pass

    @_abc.abstractmethod
    def import_weights(self, weights: _Sequence[float], i: int = 0) -> int:
        pass
