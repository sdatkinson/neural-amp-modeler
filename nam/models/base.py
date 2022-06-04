# File: base.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Lightning stuff
"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .._core import InitializableFromConfig
from .conv_net import ConvNet
from .hyper_net import HyperConvNet
from .linear import Linear


class Model(pl.LightningModule, InitializableFromConfig):
    def __init__(
        self,
        net,
        optimizer_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
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
        config = super().parse_config(config)
        net_config = config["net"]
        net = {
            "ConvNet": ConvNet.init_from_config,
            "HyperConvNet": HyperConvNet.init_from_config,
            "Linear": Linear.init_from_config,
        }[net_config["name"]](net_config["config"])
        return {
            "net": net,
            "optimizer_config": config["optimizer"],
            "scheduler_config": config["lr_scheduler"],
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

    def _shared_step(self, batch):
        args, targets = batch[:-1], batch[-1]
        preds = self(*args, pad_start=False)
        return nn.MSELoss()(preds, targets)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch)

    def validation_step(self, batch, batch_idx):
        val_loss = self._shared_step(batch)
        self.log_dict({"val_loss": val_loss})
        return val_loss
