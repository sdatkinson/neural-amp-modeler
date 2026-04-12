from typing import Any as _Any
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Sequence as _Sequence
from typing import Tuple as _Tuple

import torch as _torch
import torch.nn as _nn
from pydantic import BaseModel as _BaseModel
from pydantic import PositiveInt as _PositiveInt

from ..._core import InitializableFromConfig as _InitializableFromConfig
from .._abc import ImportsWeights as _ImportsWeights
from .._activations import PairBlend as _PairBlend
from .._activations import PairingActivation as _PairingActivation
from .._activations import PairMultiply as _PairMultiply
from .._activations import export_activation_config as _export_activation_config
from .._activations import get_activation as _get_activation
from . import _conv
from ._conv import InputMixer as _InputMixer
from ._conv import class_set as _basic_class_set
from ._film import FiLM as _FiLM
from ._slimmable import SLIMMABLE_METHOD as _SLIMMABLE_METHOD
from ._slimmable import Slimmable as _Slimmable
from ._slimmable_conv import SlimmableConv1dBase as _SlimmableConv1dBase
from ._slimmable_conv import class_set as _slimmable_class_set


class _Head1x1Config(_BaseModel):
    active: bool = False
    # NOTE: NeuralAmpModelerCore requires non-null values for out_channels and groups,
    # even though they are not used if active is False.
    out_channels: int = 1
    groups: int = 1


class _Layer1x1Config(_BaseModel):
    active: bool = True
    groups: int = 1


class _LayerArrayHeadConfig(_BaseModel):
    """Per layer-array head rechannel (``layers[i].head`` in export)."""

    out_channels: _PositiveInt
    kernel_size: _PositiveInt
    bias: bool


class _FiLMParamsConfig(_BaseModel):
    active: bool = False
    shift: bool = True
    groups: int = 1


_FILM_NAMES = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


def film_params_from_dict(d: _Optional[_Dict]) -> _FiLMParamsConfig:
    """Parse FiLM params from config dict (e.g. from .nam or init)."""
    return (
        _FiLMParamsConfig.model_validate(d)
        if d is not None
        else _FiLMParamsConfig(active=False)
    )


class _ConvFactorySet:
    """Holds conv factory callables that inject allowed_* when constructing layers."""

    def __init__(
        self,
        RechannelIn: _Any,
        LayerConv: _Any,
        InputMixer: _Any,
        Layer1x1: _Any,
        Head1x1: _Any,
        HeadRechannel: _Any,
    ):
        self.RechannelIn = RechannelIn
        self.LayerConv = LayerConv
        self.InputMixer = InputMixer
        self.Layer1x1 = Layer1x1
        self.Head1x1 = Head1x1
        self.HeadRechannel = HeadRechannel


def _get_conv_class_set(slimmable_config: _Optional[dict]) -> _conv.ClassSet:
    if slimmable_config is None:
        return _basic_class_set
    if slimmable_config.get("method") == _SLIMMABLE_METHOD:
        return _slimmable_class_set
    else:
        raise NotImplementedError(
            f"Slimmable config only supports method '{_SLIMMABLE_METHOD}', "
            f"got {slimmable_config.get('method', 'missing')!r}"
        )


def _get_conv_factory_set(
    slimmable_config: _Optional[dict],
    *,
    is_first: bool,
    is_last: bool,
    input_size: int,
    condition_size: int,
    channels: int,
    bottleneck: int,
    head_out_channels: int,
    head_rechannel_in_channels: int,
) -> _conv.ClassSet | _ConvFactorySet:
    """
    Return a conv set (classes or factories) for constructing conv layers.
    When allowed_channels is in slimmable kwargs, returns factories that inject
    allowed_in_channels/allowed_out_channels. Otherwise returns the class set.
    """
    conv_class_set = _get_conv_class_set(slimmable_config)
    allowed = (
        slimmable_config.get("kwargs", {}).get("allowed_channels")
        if slimmable_config
        else None
    )
    if allowed is None:
        return conv_class_set

    # allowed_channels provided: must be single layer array for now
    if not is_first or not is_last:
        raise ValueError(
            "allowed_channels is only supported when is_first and is_last "
            "(single layer array); multi-array support is not yet implemented"
        )
    allowed_tuple = tuple(allowed)
    slimmable_kwargs = slimmable_config.get("kwargs", {}) if slimmable_config else {}
    boosting = slimmable_kwargs.get("boosting", False)
    init_strategy = slimmable_kwargs.get("init_strategy")

    cls = conv_class_set

    # RechannelIn: is_first → allowed_in=[input_size], allowed_out=allowed_tuple
    # Slimmable base validates allowed_* vs max channels; raises _AllowedChannelsValueError
    def rechannel_in_factory(
        in_ch: int, out_ch: int, *args: _Any, is_first: bool = True, **kwargs: _Any
    ) -> _Any:
        assert issubclass(cls.RechannelIn, _SlimmableConv1dBase)
        allowed_in = (in_ch,) if is_first else allowed_tuple
        allowed_out = allowed_tuple
        return cls.RechannelIn(
            in_ch,
            out_ch,
            *args,
            allowed_in_channels=allowed_in,
            allowed_out_channels=allowed_out,
            is_first=is_first,
            boosting=boosting,
            init_strategy=init_strategy,
            **kwargs,
        )

    # LayerConv: allowed_in=allowed_tuple; paired → allowed_out=2*allowed_in
    # Slimmable base validates; raises _AllowedChannelsValueError if invalid
    def layer_conv_factory(
        in_ch: int,
        out_ch: int,
        *args: _Any,
        output_paired: bool = False,
        **kwargs: _Any,
    ) -> _Any:
        assert issubclass(cls.LayerConv, _SlimmableConv1dBase)
        allowed_in = allowed_tuple
        allowed_out = (
            tuple(2 * c for c in allowed_tuple) if output_paired else allowed_tuple
        )
        return cls.LayerConv(
            in_ch,
            out_ch,
            *args,
            allowed_in_channels=allowed_in,
            allowed_out_channels=allowed_out,
            output_paired=output_paired,
            boosting=boosting,
            init_strategy=init_strategy,
            **kwargs,
        )

    # InputMixer: allowed_in=[condition_size], allowed_out like LayerConv
    def input_mixer_factory(
        in_ch: int,
        out_ch: int,
        *args: _Any,
        output_paired: bool = False,
        **kwargs: _Any,
    ) -> _Any:
        assert issubclass(cls.InputMixer, _SlimmableConv1dBase)
        allowed_in = (in_ch,)
        allowed_out = (
            tuple(2 * c for c in allowed_tuple) if output_paired else allowed_tuple
        )
        return cls.InputMixer(
            in_ch,
            out_ch,
            *args,
            allowed_in_channels=allowed_in,
            allowed_out_channels=allowed_out,
            output_paired=output_paired,
            boosting=boosting,
            init_strategy=init_strategy,
            **kwargs,
        )

    # Layer1x1: in and out use same allowed; base validates vs in_channels, out_channels
    def layer1x1_factory(in_ch: int, out_ch: int, *args: _Any, **kwargs: _Any) -> _Any:
        allowed_both = allowed_tuple
        assert issubclass(cls.Layer1x1, _SlimmableConv1dBase)
        return cls.Layer1x1(
            in_ch,
            out_ch,
            *args,
            allowed_in_channels=allowed_both,
            allowed_out_channels=allowed_both,
            boosting=boosting,
            init_strategy=init_strategy,
            **kwargs,
        )

    # Head1x1: not fully implemented for slimmable; pass class through (no allowed)
    def head1x1_factory(in_ch: int, out_ch: int, *args: _Any, **kwargs: _Any) -> _Any:
        return cls.Head1x1(in_ch, out_ch, *args, **kwargs)

    # HeadRechannel: is_last → allowed_out=[head_out_channels], else allowed_out=allowed_tuple
    def head_rechannel_factory(
        in_ch: int, out_ch: int, *args: _Any, is_last: bool = False, **kwargs: _Any
    ) -> _Any:
        assert issubclass(cls.HeadRechannel, _SlimmableConv1dBase)
        allowed_in = allowed_tuple
        allowed_out = (out_ch,) if is_last else allowed_tuple
        return cls.HeadRechannel(
            in_ch,
            out_ch,
            *args,
            allowed_in_channels=allowed_in,
            allowed_out_channels=allowed_out,
            is_last=is_last,
            boosting=boosting,
            init_strategy=init_strategy,
            **kwargs,
        )

    return _ConvFactorySet(
        RechannelIn=rechannel_in_factory,
        LayerConv=layer_conv_factory,
        InputMixer=input_mixer_factory,
        Layer1x1=layer1x1_factory,
        Head1x1=head1x1_factory,
        HeadRechannel=head_rechannel_factory,
    )


class _Layer(_nn.Module, _InitializableFromConfig, _ImportsWeights):
    def __init__(
        self,
        conv: _conv.LayerConv,
        input_mixer: _conv.InputMixer,
        activation: _nn.Module,
        layer1x1: _conv.Conv1d,
        head1x1: _conv.Conv1d,
        conv_pre_film: _Optional[_FiLM],
        conv_post_film: _Optional[_FiLM],
        input_mixin_pre_film: _Optional[_FiLM],
        input_mixin_post_film: _Optional[_FiLM],
        activation_pre_film: _Optional[_FiLM],
        activation_post_film: _Optional[_FiLM],
        layer1x1_post_film: _Optional[_FiLM],
        head1x1_post_film: _Optional[_FiLM],
    ):
        super().__init__()
        self._conv = conv
        self._input_mixer = input_mixer
        self._activation = activation
        self._layer1x1 = layer1x1
        self._head1x1 = head1x1
        self._conv_pre_film = conv_pre_film
        self._conv_post_film = conv_post_film
        self._input_mixin_pre_film = input_mixin_pre_film
        self._input_mixin_post_film = input_mixin_post_film
        self._activation_pre_film = activation_pre_film
        self._activation_post_film = activation_post_film
        self._layer1x1_post_film = layer1x1_post_film
        self._head1x1_post_film = head1x1_post_film

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = super().parse_config(config)
        dilation = config.pop("dilations")
        channels = config.pop("channels")
        condition_size = config.pop("condition_size")
        bottleneck = config.pop("bottleneck", channels)
        kernel_size = config.pop("kernel_size")
        activation: _nn.Module = config.pop("activation")

        film_params = config.pop("film_params", dict())
        film_configs = {
            name: _FiLMParamsConfig.model_validate(film_params.get(name, {}))
            for name in _FILM_NAMES
        }

        head_1x1_config = _Head1x1Config.model_validate(
            config.pop("head_1x1_config", dict())
        )
        layer_1x1_config = _Layer1x1Config.model_validate(
            config.pop("layer_1x1_config", dict())
        )
        groups_input = config.pop("groups_input", 1)
        groups_input_mixin = config.pop("groups_input_mixin", 1)
        conv_factory_set = config.pop("conv_factory_set")

        # Input mixer takes care of the bias
        mid_channels = (
            2 * bottleneck if isinstance(activation, _PairingActivation) else bottleneck
        )
        pairing_activation = isinstance(activation, _PairingActivation)

        conv = conv_factory_set.LayerConv(
            channels,
            mid_channels,
            kernel_size,
            dilation=dilation,
            groups=groups_input,
            output_paired=pairing_activation,
        )
        input_mixer = conv_factory_set.InputMixer(
            condition_size,
            mid_channels,
            1,
            bias=False,
            groups=groups_input_mixin,
            output_paired=pairing_activation,
        )

        layer1x1 = (
            None
            if not layer_1x1_config.active
            else conv_factory_set.Layer1x1(
                bottleneck, channels, 1, groups=layer_1x1_config.groups
            )
        )
        head1x1 = (
            None
            if not head_1x1_config.active
            else conv_factory_set.Head1x1(
                bottleneck,
                head_1x1_config.out_channels,
                1,
                groups=head_1x1_config.groups,
            )
        )

        # FiLM modules (optional at each position)
        def maybe_film(
            fp: _FiLMParamsConfig, cond_dim: int, in_dim: int
        ) -> _Optional[_FiLM]:
            if not fp.active:
                return None
            return _FiLM(
                condition_size=cond_dim,
                input_dim=in_dim,
                shift=fp.shift,
                groups=fp.groups,
            )

        if film_configs["layer1x1_post_film"].active and not layer_1x1_config.active:
            raise ValueError(
                "layer1x1_post_film cannot be active when layer1x1 is not active"
            )
        if film_configs["head1x1_post_film"].active and not head_1x1_config.active:
            raise ValueError(
                "head1x1_post_film cannot be active when head1x1 is not active"
            )

        conv_pre_film = maybe_film(
            film_configs["conv_pre_film"], condition_size, channels
        )
        conv_post_film = maybe_film(
            film_configs["conv_post_film"], condition_size, mid_channels
        )
        input_mixin_pre_film = maybe_film(
            film_configs["input_mixin_pre_film"], condition_size, condition_size
        )
        input_mixin_post_film = maybe_film(
            film_configs["input_mixin_post_film"], condition_size, mid_channels
        )
        activation_pre_film = maybe_film(
            film_configs["activation_pre_film"], condition_size, mid_channels
        )
        activation_post_film = maybe_film(
            film_configs["activation_post_film"], condition_size, bottleneck
        )
        layer1x1_post_film = (
            maybe_film(film_configs["layer1x1_post_film"], condition_size, channels)
            if layer_1x1_config.active
            else None
        )
        head1x1_post_film = (
            maybe_film(
                film_configs["head1x1_post_film"],
                condition_size,
                head_1x1_config.out_channels if head_1x1_config.active else bottleneck,
            )
            if head_1x1_config.active
            else None
        )

        return dict(
            conv=conv,
            input_mixer=input_mixer,
            activation=activation,
            layer1x1=layer1x1,
            head1x1=head1x1,
            conv_pre_film=conv_pre_film,
            conv_post_film=conv_post_film,
            input_mixin_pre_film=input_mixin_pre_film,
            input_mixin_post_film=input_mixin_post_film,
            activation_pre_film=activation_pre_film,
            activation_post_film=activation_post_film,
            layer1x1_post_film=layer1x1_post_film,
            head1x1_post_film=head1x1_post_film,
        )

    @property
    def activation_name(self) -> str:
        if isinstance(self._activation, _PairingActivation):
            return self._activation.name
        else:
            return self._activation.__class__.__name__

    @property
    def bottleneck(self) -> int:
        """
        The number of channels after the activation
        """
        return (
            self.conv.out_channels // 2
            if isinstance(self._activation, _PairingActivation)
            else self.conv.out_channels
        )

    @property
    def conv_pre_film(self) -> _Optional[_FiLM]:
        return self._conv_pre_film

    @property
    def conv_post_film(self) -> _Optional[_FiLM]:
        return self._conv_post_film

    @property
    def input_mixin_pre_film(self) -> _Optional[_FiLM]:
        return self._input_mixin_pre_film

    @property
    def input_mixin_post_film(self) -> _Optional[_FiLM]:
        return self._input_mixin_post_film

    @property
    def activation_pre_film(self) -> _Optional[_FiLM]:
        return self._activation_pre_film

    @property
    def activation_post_film(self) -> _Optional[_FiLM]:
        return self._activation_post_film

    @property
    def layer1x1_post_film(self) -> _Optional[_FiLM]:
        return self._layer1x1_post_film

    @property
    def head1x1_post_film(self) -> _Optional[_FiLM]:
        return self._head1x1_post_film

    @property
    def channels(self) -> int:
        return self.conv.in_channels

    @property
    def conv(self) -> _conv.Conv1d:
        return self._conv

    @property
    def dilation(self) -> int:
        return self.conv.dilation[0]

    @property
    def head1x1(self) -> _conv.Conv1d | None:
        return self._head1x1

    @property
    def input_mixer(self) -> _InputMixer:
        return self._input_mixer

    @property
    def kernel_size(self) -> int:
        return self._conv.kernel_size[0]

    @property
    def layer1x1(self) -> _conv.Conv1d | None:
        return self._layer1x1

    def export_activation_config(self):
        """
        Export activation in WaveNet Factory format: (primary, gating_mode, secondary).
        Parses the output of _activations.export_activation_config() into primary/
        secondary/gating_mode as expected by the C++ Factory.
        """
        out = _export_activation_config(self._activation)
        if isinstance(self._activation, _PairingActivation):
            if isinstance(self._activation, _PairMultiply):
                gating_mode = "gated"
            elif isinstance(self._activation, _PairBlend):
                gating_mode = "blended"
            else:
                raise ValueError(f"Unknown pairing activation: {self._activation}")
            return {
                "primary": out["primary"],
                "gating_mode": gating_mode,
                "secondary": out["secondary"],
            }
        else:
            return {
                "primary": out,
                "gating_mode": "none",
                "secondary": None,
            }

    def export_weights(self) -> _torch.Tensor:
        tensors = [
            self.conv.export_weights(),
            self._input_mixer.export_weights(),
        ]
        if self._layer1x1 is not None:
            tensors.append(self._layer1x1.export_weights())
        if self.head1x1 is not None:
            tensors.append(self.head1x1.export_weights())
        for film in (
            self._conv_pre_film,
            self._conv_post_film,
            self._input_mixin_pre_film,
            self._input_mixin_post_film,
            self._activation_pre_film,
            self._activation_post_film,
            self._layer1x1_post_film,
            self._head1x1_post_film,
        ):
            if film is not None:
                tensors.append(film.export_weights())
        return _torch.cat(tensors)

    def forward(
        self, x: _torch.Tensor, h: _torch.Tensor, out_length: int
    ) -> _Tuple[_Optional[_torch.Tensor], _torch.Tensor]:
        """
        :param x: (B,C,L1) From last layer
        :param h: (B,DX,L2) Conditioning. If first, ignored.

        :return:
            If not final:
                (B,C,L1-d) to next layer
                (B,C,L1-d) to mixer
            If final, next layer is None
        """

        # Helper: slice condition to match tensor time length (conv shortens sequence)
        def _c(t_len: int, tensor: _torch.Tensor = h) -> _torch.Tensor:
            return tensor[:, :, -t_len:]

        # Step 1: input convolution (with optional pre/post FiLM)
        conv_input = x
        if self._conv_pre_film is not None:
            pre_conv_length = min(conv_input.shape[2], h.shape[2])
            conv_input = self._conv_pre_film(
                _c(pre_conv_length, tensor=conv_input), _c(pre_conv_length)
            )
        zconv = self.conv(conv_input)
        if self._conv_post_film is not None:
            post_conv_length = min(zconv.shape[2], h.shape[2])
            zconv = self._conv_post_film(
                _c(post_conv_length, tensor=zconv), _c(post_conv_length)
            )

        # Input mixin (with optional pre/post FiLM)
        mixin_input = h
        if self._input_mixin_pre_film is not None:
            mixin_input = self._input_mixin_pre_film(mixin_input, h)
        mix_out = self._input_mixer(mixin_input)[:, :, -zconv.shape[2] :]
        if self._input_mixin_post_film is not None:
            mix_out = self._input_mixin_post_film(mix_out, _c(mix_out.shape[2]))

        z1len = min(zconv.shape[2], mix_out.shape[2])
        z1 = zconv[:, :, -z1len:] + mix_out[:, :, -z1len:]
        if self._activation_pre_film is not None:
            z1 = self._activation_pre_film(z1, _c(z1.shape[2]))

        post_activation = self._activation(z1)
        if self._activation_post_film is not None:
            post_activation = self._activation_post_film(
                post_activation, _c(post_activation.shape[2])
            )

        layer_output = post_activation
        if self._layer1x1 is not None:
            layer_output = self._layer1x1(layer_output)
            if self._layer1x1_post_film is not None:
                layer_output = self._layer1x1_post_film(
                    layer_output, _c(layer_output.shape[2])
                )

        head_output = post_activation
        if self.head1x1 is not None:
            head_output = self.head1x1(head_output)[:, :, -out_length:]
            if self._head1x1_post_film is not None:
                head_output = self._head1x1_post_film(
                    head_output, _c(head_output.shape[2])
                )
        else:
            head_output = head_output[:, :, -out_length:]

        residual = x[:, :, -layer_output.shape[2] :] + layer_output
        return (residual, head_output)

    def import_weights(self, weights: _Sequence[float], i: int) -> int:
        i = self.conv.import_weights(weights, i)
        i = self._input_mixer.import_weights(weights, i)
        if self._layer1x1 is not None:
            i = self._layer1x1.import_weights(weights, i)
        if self.head1x1 is not None:
            i = self.head1x1.import_weights(weights, i)
        for film in (
            self._conv_pre_film,
            self._conv_post_film,
            self._input_mixin_pre_film,
            self._input_mixin_post_film,
            self._activation_pre_film,
            self._activation_post_film,
            self._layer1x1_post_film,
            self._head1x1_post_film,
        ):
            if film is not None:
                i = film.import_weights(weights, i)
        return i

    def get_slimmable_config(self):
        # Quick heuristic: look at the conv.
        if isinstance(self.conv, _Slimmable):
            kwargs = {}
            if isinstance(self.conv, _SlimmableConv1dBase):
                kwargs["allowed_channels"] = list(self.conv._allowed_in_channels)
            return {"method": _SLIMMABLE_METHOD, "kwargs": kwargs}


class LayerArray(_nn.Module, _InitializableFromConfig):
    """
    Takes in the input and condition (and maybe the head input so far); outputs the
    layer output and head input.

    The original WaveNet only uses one of these, but you can stack multiple of this
    module to vary the channels throughout with minimal extra channel-changing conv
    layers.
    """

    def __init__(
        self,
        rechannel: _conv.RechannelIn,
        layers: _nn.ModuleList,
        head_rechannel: _conv.HeadRechannel,
    ):
        super().__init__()

        self._rechannel = rechannel
        self._layers = layers
        self._head_rechannel = head_rechannel

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = super().parse_config(config)

        is_first = config.pop("is_first")
        is_last = config.pop("is_last")
        input_size = config.pop("input_size")
        condition_size = config.pop("condition_size")
        if "head" not in config:
            raise KeyError(
                "Each layer array config must include a 'head' object with "
                "'out_channels', 'kernel_size', and 'bias'"
            )
        head_cfg = _LayerArrayHeadConfig.model_validate(config.pop("head"))
        channels = config.pop("channels")
        if "kernel_sizes" in config:
            kernel_sizes = config.pop("kernel_sizes")
        elif "kernel_size" in config:
            kernel_sizes = config.pop("kernel_size")
        else:
            raise KeyError("Either 'kernel_sizes' or 'kernel_size' must be present")
        dilations = config.pop("dilations")
        activation = config.pop("activation")
        bottleneck = config.pop("bottleneck", channels)

        head1x1_config = _Head1x1Config.model_validate(
            config.pop("head_1x1_config", dict())
        )
        layer1x1_config = _Layer1x1Config.model_validate(
            config.pop("layer_1x1_config", dict())
        )
        film_params = config.pop("film_params", dict())
        groups_input = config.pop("groups_input", 1)
        groups_input_mixin = config.pop("groups_input_mixin", 1)
        slimmable_config = config.pop("slimmable", None)

        if slimmable_config is not None and head_cfg.kernel_size != 1:
            raise NotImplementedError(
                "Slimmable training with head rechannel kernel_size != 1 is not supported"
            )

        head_rechannel_in_channels = (
            head1x1_config.out_channels if head1x1_config.active else bottleneck
        )
        conv_factory_set = _get_conv_factory_set(
            slimmable_config,
            is_first=is_first,
            is_last=is_last,
            input_size=input_size,
            condition_size=condition_size,
            channels=channels,
            bottleneck=bottleneck,
            head_out_channels=head_cfg.out_channels,
            head_rechannel_in_channels=head_rechannel_in_channels,
        )

        rechannel = conv_factory_set.RechannelIn(
            input_size, channels, 1, bias=False, is_first=is_first
        )

        num_layers = len(dilations)

        # Broadcast configs to all layers:
        # Activation:
        if isinstance(activation, (str, dict)):
            activation = [activation] * num_layers
        else:
            assert isinstance(
                activation, _Sequence
            ), "activation must be a string, dict, or sequence"
        a_list = [_get_activation(a) for a in activation]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        assert isinstance(
            kernel_sizes, _Sequence
        ), "kernel_sizes must be a int or sequence"
        assert (
            len(kernel_sizes) == num_layers
        ), "kernel_sizes must be the same length as dilations"

        layers = _nn.ModuleList(
            [
                _Layer.init_from_config(
                    {
                        "condition_size": condition_size,
                        "channels": channels,
                        "kernel_size": k,
                        "dilations": d,
                        "activation": a,
                        "bottleneck": bottleneck,
                        "head_1x1_config": head1x1_config,
                        "layer_1x1_config": layer1x1_config,
                        "film_params": film_params,
                        "groups_input": groups_input,
                        "groups_input_mixin": groups_input_mixin,
                        "slimmable": slimmable_config,
                        "conv_factory_set": conv_factory_set,
                    }
                )
                for k, d, a in zip(kernel_sizes, dilations, a_list)
            ]
        )
        head_rechannel = conv_factory_set.HeadRechannel(
            head_rechannel_in_channels,
            head_cfg.out_channels,
            head_cfg.kernel_size,
            bias=head_cfg.bias,
            is_last=is_last,
        )

        return dict(
            rechannel=rechannel,
            layers=layers,
            head_rechannel=head_rechannel,
        )

    @property
    def receptive_field(self) -> int:
        return (
            self._receptive_field_no_head_rechannel
            + int(self._head_rechannel.kernel_size[0])
            - 1
        )

    @property
    def head_channels(self) -> int:
        """
        Channel width of the head path after this layer array (output of
        ``head_rechannel``), i.e. ``head.out_channels`` in config.
        """
        return self._head_rechannel.out_channels

    def export_config(self):
        # Use first layer for things that are assumed to be constant across layers.
        first_layer = self._layers[0]
        assert isinstance(first_layer, _Layer)

        head1x1_config = (
            _Head1x1Config()
            if first_layer.head1x1 is None
            else _Head1x1Config(
                active=True,
                out_channels=first_layer.head1x1.out_channels,
                groups=first_layer.head1x1.groups,
            )
        )
        layer1x1_config = (
            _Layer1x1Config(active=False)
            if first_layer.layer1x1 is None
            else _Layer1x1Config(active=True, groups=first_layer.layer1x1.groups)
        )

        def get_film_params(key):
            film = getattr(first_layer, key)
            return (
                _FiLMParamsConfig(active=False, shift=True, groups=1).model_dump()
                if film is None
                else _FiLMParamsConfig(
                    active=True, shift=film.shift, groups=film.groups
                ).model_dump()
            )

        activations = []
        gating_modes = []
        secondary_activations = []
        for layer in self._layers:
            activation_config = layer.export_activation_config()
            activations.append(activation_config["primary"])
            gating_modes.append(activation_config["gating_mode"])
            secondary_activations.append(activation_config["secondary"])

        config = {
            "input_size": self._rechannel.in_channels,
            "condition_size": first_layer.input_mixer.in_channels,
            "head": {
                "out_channels": self._head_rechannel.out_channels,
                "kernel_size": int(self._head_rechannel.kernel_size[0]),
                "bias": self._head_rechannel.bias is not None,
            },
            "channels": first_layer.channels,
            "kernel_sizes": [layer.kernel_size for layer in self._layers],
            "dilations": self._dilations,
            "activation": activations,
            "bottleneck": first_layer.bottleneck,
            "head1x1": head1x1_config.model_dump(),
            "layer1x1": layer1x1_config.model_dump(),
            "groups_input": first_layer.conv.groups,
            "groups_input_mixin": first_layer.input_mixer.groups,
            # FiLM params:
            **{key: get_film_params(key) for key in _FILM_NAMES},
            "gating_mode": gating_modes,
            "secondary_activation": secondary_activations,
            "slimmable": first_layer.get_slimmable_config(),
        }
        return config

    def export_weights(self) -> _torch.Tensor:
        assert all(isinstance(layer, _Layer) for layer in self._layers)
        # Help typing:
        layer_weights = []
        for layer in self._layers:
            assert isinstance(layer, _Layer)
            layer_weights.append(layer.export_weights())

        return _torch.cat(
            [self._rechannel.export_weights()]
            + layer_weights
            + [self._head_rechannel.export_weights()]
        )

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        i = self._rechannel.import_weights(weights, i)
        for layer in self._layers:
            i = layer.import_weights(weights, i)
        return self._head_rechannel.import_weights(weights, i)

    def forward(
        self,
        x: _torch.Tensor,
        c: _torch.Tensor,
        head_input: _Optional[_torch.Tensor] = None,
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """
        :param x: (B,Dx,L) layer input
        :param c: (B,Dc,L) condition

        Layer array receptive field is R.

        :return:
            (B,Dc,L-(R-1)) head input
            (B,Dc,L-(R-1)) layer output
        """
        out_length = min(x.shape[2], c.shape[2]) - (self.receptive_field - 1)
        out_length_no_head_rechannel = min(x.shape[2], c.shape[2]) - (
            self._receptive_field_no_head_rechannel - 1
        )
        x = self._rechannel(x)
        for layer in self._layers:
            x, head_term = layer(
                x, c, out_length_no_head_rechannel
            )  # Ensures head_term sample length
            head_input = (
                head_term
                if head_input is None
                else head_input[:, :, -out_length_no_head_rechannel:] + head_term
            )
        head_rechannel_output = self._head_rechannel(head_input)

        # Trim x to match loss of sequence length by head_rechannel (valid conv).
        assert head_rechannel_output.shape[2] == out_length
        # Residual length can differ from out_length_no_head_rechannel when c is shorter
        # than x (e.g. condition_dsp): layers min along the condition path.
        assert x.shape[2] >= out_length
        x = x[:, :, -out_length:]

        return head_rechannel_output, x

    @property
    def _dilations(self) -> _Sequence[int]:
        dilations = []
        for layer in self._layers:
            assert isinstance(layer, _Layer)
            dilations.append(layer.dilation)
        return dilations

    @property
    def _receptive_field_no_head_rechannel(self) -> int:
        total = 1
        for layer in self._layers:
            assert isinstance(layer, _Layer)
            total += (layer.kernel_size - 1) * layer.dilation
        return total

    @classmethod
    def _parse_slimmable_config(cls, val: _Any) -> bool:
        """
        Parse slimmable config from layer config.
        Only {"method": "slice_channels_uniform", "kwargs": {...}} is supported.
        Raises NotImplementedError for any other value.
        """
        # In the future, it may be necessary to have sight on the slimmable configs of
        # the preceding and following layer arrays (e.g. if they slim faster or slower,
        # do it in some other way, etc)
        if val is None:
            return False
        if not isinstance(val, dict):
            raise NotImplementedError(
                f"Slimmable config must be a dict, got {type(val).__name__}"
            )
        if val.get("method") != _SLIMMABLE_METHOD:
            raise NotImplementedError(
                f"Slimmable config only supports method '{_SLIMMABLE_METHOD}', "
                f"got {val.get('method', 'missing')!r}"
            )
        if "kwargs" not in val:
            raise NotImplementedError("Slimmable config must include 'kwargs' key")
        if not isinstance(val["kwargs"], dict):
            raise NotImplementedError(
                f"Slimmable config 'kwargs' must be a dict, got {type(val['kwargs']).__name__}"
            )
        return True
