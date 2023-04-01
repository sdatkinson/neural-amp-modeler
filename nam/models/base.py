# File: base.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Implements the base PyTorch Lightning model.
This is meant to combine an acutal model (subclassed from `._base.BaseNet` or 
`._base.ParametricBaseNet`) along with loss function boilerplate.

For the base *PyTorch* model containing the actual architecture, see `._base`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import auraloss
import logging
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .._core import InitializableFromConfig
from .conv_net import ConvNet
from .linear import Linear
from .losses import esr, multi_resolution_stft_loss, mse_fft
from .parametric.catnets import CatLSTM, CatWaveNet
from .parametric.hyper_net import HyperConvNet
from .recurrent import LSTM
from .wavenet import WaveNet

logger = logging.getLogger(__name__)


class ValidationLoss(Enum):
    """
    mse: mean squared error
    esr: error signal ratio (Eq. (10) from
        https://www.mdpi.com/2076-3417/10/3/766/htm
        NOTE: Be careful when computing ESR on minibatches! The average ESR over
        a minibatch of data not the same as the ESR of all of the same data in
        the minibatch calculated over at once (because of the denominator).
        (Hint: think about what happens if one item in the minibatch is all
        zeroes...)
    """

    MSE = "mse"
    ESR = "esr"


@dataclass
class LossConfig(InitializableFromConfig):
    """
    :param mask_first: How many of the first samples to ignore when comptuing the loss.
    :param dc_weight: Weight for the DC loss term. If 0, ignored.
    :params val_loss: Which loss to track for the best model checkpoint.
    :param pre_emph_coef: Coefficient of 1st-order pre-emphasis filter from
        https://www.mdpi.com/2076-3417/10/3/766. Paper value: 0.95.
    """

    mrstft_weight: float = 0.0  # 0.0 means no multiresolution stft loss, 2e-4 works pretty well if one wants to use it
    fourier: bool = False
    mask_first: int = 0
    dc_weight: float = 0.0
    val_loss: ValidationLoss = ValidationLoss.MSE
    pre_emph_weight: Optional[float] = None
    pre_emph_coef: Optional[float] = None

    @classmethod
    def parse_config(cls, config):
        config = super().parse_config(config)
        fourier = config.get("fourier", False)
        dc_weight = config.get("dc_weight", 0.0)
        val_loss = ValidationLoss(config.get("val_loss", "mse"))
        mask_first = config.get("mask_first", 0)
        pre_emph_coef = config.get("pre_emph_coef")
        pre_emph_weight = config.get("pre_emph_weight")
        mrstft_weight = config.get("mstft_weight", 0.0)
        return {
            "fourier": fourier,
            "mask_first": mask_first,
            "dc_weight": dc_weight,
            "val_loss": val_loss,
            "pre_emph_coef": pre_emph_coef,
            "pre_emph_weight": pre_emph_weight,
            "mrstft_weight": mrstft_weight,
        }

    def apply_mask(self, *args):
        """
        :param args: (L,) or (B,)
        :return: (L-M,) or (B, L-M)
        """
        return tuple(a[..., self.mask_first :] for a in args)


class Model(pl.LightningModule, InitializableFromConfig):
    def __init__(
        self,
        net,
        optimizer_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        loss_config: Optional[LossConfig] = None,
    ):
        """
        :param scheduler_config: contains
            Required:
            * "class"
            * "kwargs"
            Optional (defaults to Lightning defaults):
            * "interval" ("epoch" of "step")
            * "frequency" (int)
            * "monitor" (str)
        """
        super().__init__()
        self._net = net
        self._optimizer_config = {} if optimizer_config is None else optimizer_config
        self._scheduler_config = scheduler_config
        self._loss_config = LossConfig() if loss_config is None else loss_config
        self._mrstft = None  # Multi-resolution short-time Fourier transform loss
        # Where to compute the MRSTFT.
        # Keeping it on-device is preferable, but if that fails, then remember to drop
        # it to cpu from then on.
        self._mrstft_device: Optional[torch.device] = None

    @classmethod
    def init_from_config(cls, config):
        checkpoint_path = config.get("checkpoint_path")
        config = cls.parse_config(config)
        return (
            cls(**config)
            if checkpoint_path is None
            else cls.load_from_checkpoint(checkpoint_path, **config)
        )

    @classmethod
    def parse_config(cls, config):
        """
        e.g.

        {
            "net": {
                "name": "ConvNet",
                "config": {...}
            },
            "loss": {
                "dc_weight": 0.1
            },
            "optimizer": {
                "lr": 0.0003
            },
            "lr_scheduler": {
                "class": "ReduceLROnPlateau",
                "kwargs": {
                    "factor": 0.8,
                    "patience": 10,
                    "cooldown": 15,
                    "min_lr": 1e-06,
                    "verbose": true
                },
                "monitor": "val_loss"
            }
        }
        """
        config = super().parse_config(config)
        net_config = config["net"]
        net = {
            "CatLSTM": CatLSTM.init_from_config,
            "CatWaveNet": CatWaveNet.init_from_config,
            "ConvNet": ConvNet.init_from_config,
            "HyperConvNet": HyperConvNet.init_from_config,
            "Linear": Linear.init_from_config,
            "LSTM": LSTM.init_from_config,
            "WaveNet": WaveNet.init_from_config,
        }[net_config["name"]](net_config["config"])
        loss_config = LossConfig.init_from_config(config.get("loss", {}))
        return {
            "net": net,
            "optimizer_config": config["optimizer"],
            "scheduler_config": config["lr_scheduler"],
            "loss_config": loss_config,
        }

    @property
    def net(self) -> nn.Module:
        return self._net

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self._optimizer_config)
        if self._scheduler_config is None:
            return optimizer
        else:
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, self._scheduler_config["class"]
            )(optimizer, **self._scheduler_config["kwargs"])
            lr_scheduler_config = {"scheduler": lr_scheduler}
            for key in ("interval", "frequency", "monitor"):
                if key in self._scheduler_config:
                    lr_scheduler_config[key] = self._scheduler_config[key]
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def _shared_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        B: Batch size
        L: Sequence length

        :return: (B,L), (B,L)
        """
        args, targets = batch[:-1], batch[-1]
        preds = self(*args, pad_start=False)

        return preds, targets

    def training_step(self, batch, batch_idx):
        preds, targets = self._shared_step(batch)

        loss = 0.0
        # Prediction aka MSE loss
        if self._loss_config.fourier:
            loss = loss + mse_fft(preds, targets)
        else:
            loss = loss + self._mse_loss(preds, targets)
        if self._loss_config.mrstft_weight > 0.0:
            loss = loss + self._loss_config.mrstft_weight * self._mrstft_loss(
                preds, targets
            )
        # Pre-emphasized MSE
        if self._loss_config.pre_emph_weight is not None:
            if (self._loss_config.pre_emph_coef is None) != (
                self._loss_config.pre_emph_weight is None
            ):
                raise ValueError("Invalid pre-emph")
            loss = loss + self._loss_config.pre_emph_weight * self._mse_loss(
                preds, targets, pre_emph_coef=self._loss_config.pre_emph_coef
            )

        # DC loss
        dc_weight = self._loss_config.dc_weight
        if dc_weight > 0.0:
            # Denominator could be a bad idea. I'm going to omit it esp since I'm
            # using mini batches
            mean_dims = torch.arange(1, preds.ndim).tolist()
            dc_loss = nn.MSELoss()(
                preds.mean(dim=mean_dims), targets.mean(dim=mean_dims)
            )
            loss = loss + dc_weight * dc_loss
        return loss

    def validation_step(self, batch, batch_idx):
        preds, targets = self._shared_step(batch)
        mse_loss = self._mse_loss(preds, targets)
        esr_loss = self._esr_loss(preds, targets)
        val_loss = {ValidationLoss.MSE: mse_loss, ValidationLoss.ESR: esr_loss}[
            self._loss_config.val_loss
        ]
        dict_to_log = {"MSE": mse_loss, "ESR": esr_loss, "val_loss": val_loss}
        if self._loss_config.mrstft_weight > 0.0 and self._mrstft is not None:
            mrstft_loss = self._mrstft_loss(preds, targets)
            dict_to_log.update({"MRSTFT": mrstft_loss})
        self.log_dict(dict_to_log)
        return val_loss

    def _esr_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Error signal ratio aka ESR loss.

        Eq. (10), from
        https://www.mdpi.com/2076-3417/10/3/766/htm

        B: Batch size
        L: Sequence length

        :param preds: (B,L)
        :param targets: (B,L)
        :return: ()
        """
        return esr(preds, targets)

    def _mse_loss(self, preds, targets, pre_emph_coef: Optional[float] = None):
        if pre_emph_coef is not None:
            preds, targets = [
                z[..., 1:] - pre_emph_coef * z[..., :-1] for z in (preds, targets)
            ]
        return nn.MSELoss()(preds, targets)

    def _mrstft_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Experimental Multi Resolution Short Time Fourier Transform Loss using auraloss implementation.
        B: Batch size
        L: Sequence length

        :param preds: (B,L)
        :param targets: (B,L)
        :return: ()
        """
        if self._mrstft is None:
            self._mrstft = auraloss.freq.MultiResolutionSTFTLoss()

        backup_device = "cpu"

        try:
            return multi_resolution_stft_loss(
                preds, targets, self._mrstft, device=self._mrstft_device
            )
        except Exception as e:
            if self._mrstft_device == backup_device:
                raise e
            logger.warning("MRSTFT failed on device; falling back to CPU")
            self._mrstft_device = backup_device
            return multi_resolution_stft_loss(
                preds, targets, self._mrstft, device=self._mrstft_device
            )
