# File: test_lightning_module.py
# Created Date: Sunday November 24th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

from typing import Optional as _Optional

import pytest as _pytest
import torch as _torch
from auraloss.freq import MultiResolutionSTFTLoss as _MultiResolutionSTFTLoss

from nam.train import lightning_module as _lightning_module

from ..test_models.test_base import MockBaseNet as _MockBaseNet


@_pytest.mark.parametrize(
    "batch_size,sequence_length", ((16, 8192), (3, 2048), (1, 4000))
)
def test_mrstft_loss(batch_size: int, sequence_length: int):
    obj = _lightning_module.LightningModule(
        _MockBaseNet(1.0),
        loss_config=_lightning_module.LossConfig(mrstft_weight=0.0002),
    )
    preds = _torch.randn((batch_size, sequence_length))
    targets = _torch.randn(preds.shape)
    loss = obj._mrstft_loss(preds, targets)
    assert isinstance(loss, _torch.Tensor)
    assert loss.ndim == 0


def test_mrstft_loss_cpu_fallback(mocker):
    """
    Assert that fallback to CPU happens on failure

    :param mocker: Provided by pytest-mock
    """

    def mocked_loss(
        preds: _torch.Tensor,
        targets: _torch.Tensor,
        loss_func: _Optional[_MultiResolutionSTFTLoss] = None,
        device: _Optional[_torch.device] = None,
    ) -> _torch.Tensor:
        """
        As if the device doesn't support it
        """
        if device != "cpu":
            raise RuntimeError("Trigger fallback")
        return _torch.tensor(1.0)

    mocker.patch("nam.train.lightning_module._multi_resolution_stft_loss", mocked_loss)

    batch_size = 3
    sequence_length = 4096
    obj = _lightning_module.LightningModule(
        _MockBaseNet(1.0),
        loss_config=_lightning_module.LossConfig(mrstft_weight=0.0002),
    )
    preds = _torch.randn((batch_size, sequence_length))
    targets = _torch.randn(preds.shape)

    assert obj._mrstft_device is None
    obj._mrstft_loss(preds, targets)  # Should trigger fallback
    assert obj._mrstft_device == "cpu"


if __name__ == "__main__":
    _pytest.main()
