"""
Convolutional layers for WaveNet.
"""

from typing import NamedTuple as _NamedTuple
from typing import Sequence as _Sequence

import torch as _torch
import torch.nn as _nn

from .._abc import ImportsWeights as _ImportsWeights


class Conv1d(_nn.Conv1d, _ImportsWeights):
    """
    Adds in NAM functionality to export and import weights.
    """

    def export_weights(self) -> _torch.Tensor:
        tensors = []
        if self.weight is not None:
            tensors.append(self.weight.data.flatten())
        if self.bias is not None:
            tensors.append(self.bias.data.flatten())
        if len(tensors) == 0:
            return _torch.zeros((0,))
        else:
            return _torch.cat(tensors)

    def import_weights(self, weights: _Sequence[float], i: int) -> int:
        weights_tensor = _torch.Tensor(weights)
        if self.weight is not None:
            n = self.weight.numel()
            self.weight.data = (
                weights_tensor[i : i + n]
                .reshape(self.weight.shape)
                .to(self.weight.device)
            )
            i += n
        if self.bias is not None:
            n = self.bias.numel()
            self.bias.data = (
                weights_tensor[i : i + n].reshape(self.bias.shape).to(self.bias.device)
            )
            i += n
        return i


def apply_stable(conv: Conv1d) -> None:
    """Set conv bias to zero and disable grads. For use when stable=True."""
    if conv.bias is not None:
        conv.bias.data.zero_()
        conv.bias.requires_grad = False


# NOTE: we could have the __init__s (silently?) drop extra arguments. That'd
# be cleaner in terms of these classes not needing to knwo about the other
# implementations. The flip-side is that it could be dangerous in terms of hard-to-find
# bugs.
#
# For now, we'll make it kinda yucky in order to be safer.


class RechannelIn(Conv1d):
    def __init__(self, *args, is_first: bool = True, **kwargs):
        """
        :param is_first: Whether this is the first layer array in the WaveNet. If it is,
            then the input doesn't care about slimming, so you need to make sure that
            you always accept it at the same size.
        """
        super().__init__(*args, **kwargs)
        # HACK do this to drop is_first arg that slimmable needs


class LayerConv(Conv1d):
    def __init__(self, *args, output_paired: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # HACK do this to drop output_paired arg that slimmable needs


class InputMixer(Conv1d):
    def __init__(self, *args, output_paired: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # HACK do this to drop output_paired arg that slimmable needs


class HeadRechannel(Conv1d):
    def __init__(self, *args, is_last: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # HACK do this to drop output_paired arg that slimmable needs


# Create class set instances from this...
class ClassSet(_NamedTuple):
    """
    All of the convolutional layers in the WaveNet model.
    """

    RechannelIn: type[RechannelIn]
    LayerConv: type[LayerConv]
    InputMixer: type[InputMixer]
    Layer1x1: type[Conv1d]
    Head1x1: type[Conv1d]
    HeadRechannel: type[HeadRechannel]


# The basic class set (no slimmable)
class_set = ClassSet(
    RechannelIn=RechannelIn,
    LayerConv=LayerConv,
    InputMixer=InputMixer,
    Layer1x1=Conv1d,
    Head1x1=Conv1d,
    HeadRechannel=HeadRechannel,
)

# The idea is to add other ways of slimming by defining a class set for them.
# Then, the layer array uses the class set according to the slimmable configuration
# given to it.
