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


def test_get_loss_dict():
    obj = _lightning_module.LightningModule(
        _MockBaseNet(1.0),
        loss_config=_lightning_module.LossConfig(mrstft_weight=0.0002),
    )
    preds = _torch.randn((3, 4096))
    targets = _torch.randn(preds.shape)
    loss_dict = obj._get_loss_dict(preds, targets)
    assert isinstance(loss_dict, dict)
    # MSE will also be computed by default.
    assert len(loss_dict) >= 1
    assert "MRSTFT" in loss_dict
    assert loss_dict["MRSTFT"].value is not None
    assert loss_dict["MRSTFT"].weight is not None


def test_get_loss_dict_custom_loss():
    expected_key = "my_custom_loss"
    expected_weight = 0.1

    def custom_loss(preds, targets):
        return _torch.max(_torch.abs(preds - targets), dim=1).values.mean()

    obj = _lightning_module.LightningModule(
        _MockBaseNet(1.0),
        loss_config=_lightning_module.LossConfig(
            mse_weight=None,
            mrstft_weight=0.0002,
            custom_losses={
                expected_key: _lightning_module._CustomLoss(
                    weight=expected_weight, func=custom_loss
                )
            },
        ),
    )
    preds = _torch.randn((3, 4096))
    targets = _torch.randn(preds.shape)
    loss_dict = obj._get_loss_dict(preds, targets)
    assert isinstance(loss_dict, dict)
    assert len(loss_dict) == 2
    for key in ("MRSTFT", expected_key):
        assert key in loss_dict
        assert loss_dict[key].value is not None
        assert loss_dict[key].weight is not None


def test_custom_losses_init():
    """
    Assert that a custom loss can be included in the loss config.
    """
    key = "my_custom_loss"
    expected_weight = 0.1
    config = {
        "custom_losses": {
            key: {"name": "torch.nn.MSELoss", "kwargs": {}, "weight": expected_weight},
        }
    }
    config = _lightning_module.LossConfig.parse_config(config)
    custom_losses = config["custom_losses"]
    assert custom_losses is not None
    assert isinstance(custom_losses, dict)
    assert key in custom_losses
    value = custom_losses[key]
    assert isinstance(value, _lightning_module._CustomLoss)
    assert value.weight == expected_weight
    # And just make sure it runs:
    batch_size, sequence_length = 3, 5
    preds = _torch.randn((batch_size, sequence_length))
    targets = _torch.randn(preds.shape)
    loss = value.func(preds, targets)
    assert isinstance(loss, _torch.Tensor)
    assert loss.ndim == 0
    # Anc coincidentally for this test:
    assert loss.item() == _torch.nn.MSELoss()(preds, targets)


if __name__ == "__main__":
    _pytest.main()
