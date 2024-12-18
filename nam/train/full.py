# File: full.py
# Created Date: Tuesday March 26th 2024
# Author: Enrico Schifano (eraz1997@live.it)

import json as _json
from pathlib import Path as _Path
from time import time as _time
from typing import Optional as _Optional, Union as _Union
from warnings import warn as _warn

import matplotlib.pyplot as _plt
import numpy as _np
import pytorch_lightning as _pl
from pytorch_lightning.utilities.warnings import (
    PossibleUserWarning as _PossibleUserWarning,
)
import torch as _torch
from torch.utils.data import DataLoader as _DataLoader

from nam.data import (
    ConcatDataset as _ConcatDataset,
    Split as _Split,
    init_dataset as _init_dataset,
)
from nam.train.lightning_module import LightningModule as _LightningModule
from nam.util import filter_warnings as _filter_warnings

_torch.manual_seed(0)


def _rms(x: _Union[_np.ndarray, _torch.Tensor]) -> float:
    if isinstance(x, _np.ndarray):
        return _np.sqrt(_np.mean(_np.square(x)))
    elif isinstance(x, _torch.Tensor):
        return _torch.sqrt(_torch.mean(_torch.square(x))).item()
    else:
        raise TypeError(type(x))


def _plot(
    model,
    ds,
    savefig=None,
    show=True,
    window_start: _Optional[int] = None,
    window_end: _Optional[int] = None,
):
    if isinstance(ds, _ConcatDataset):

        def extend_savefig(i, savefig):
            if savefig is None:
                return None
            savefig = _Path(savefig)
            extension = savefig.name.split(".")[-1]
            stem = savefig.name[: -len(extension) - 1]
            return _Path(savefig.parent, f"{stem}_{i}.{extension}")

        for i, ds_i in enumerate(ds.datasets):
            _plot(
                model,
                ds_i,
                savefig=extend_savefig(i, savefig),
                show=show and i == len(ds.datasets) - 1,
                window_start=window_start,
                window_end=window_end,
            )
        return
    with _torch.no_grad():
        tx = len(ds.x) / 48_000
        print(f"Run (t={tx:.2f})")
        t0 = _time()
        output = model(ds.x).flatten().cpu().numpy()
        t1 = _time()
        try:
            rt = f"{tx / (t1 - t0):.2f}"
        except ZeroDivisionError as e:
            rt = "???"
        print(f"Took {t1 - t0:.2f} ({rt}x)")

    _plt.figure(figsize=(16, 5))
    _plt.plot(output[window_start:window_end], label="Prediction")
    _plt.plot(ds.y[window_start:window_end], linestyle="--", label="Target")
    nrmse = _rms(_torch.Tensor(output) - ds.y) / _rms(ds.y)
    esr = nrmse**2
    _plt.title(f"ESR={esr:.3f}")
    _plt.legend()
    if savefig is not None:
        _plt.savefig(savefig)
    if show:
        _plt.show()


def _create_callbacks(learning_config):
    """
    Checkpointing, essentially
    """
    # Checkpoints should be run every time the validation check is run.
    # So base it off of learning_config["trainer"]["val_check_interval"] if it's there.
    validate_inside_epoch = "val_check_interval" in learning_config["trainer"]
    if validate_inside_epoch:
        kwargs = {
            "every_n_train_steps": learning_config["trainer"]["val_check_interval"]
        }
    else:
        kwargs = {
            "every_n_epochs": learning_config["trainer"].get(
                "check_val_every_n_epoch", 1
            )
        }

    checkpoint_best = _pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename="{epoch:04d}_{step}_{ESR:.3e}_{MSE:.3e}",
        save_top_k=3,
        monitor="val_loss",
        **kwargs,
    )

    # return [checkpoint_best, checkpoint_last]
    # The last epoch that was finished.
    checkpoint_epoch = _pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename="checkpoint_epoch_{epoch:04d}", every_n_epochs=1
    )
    if not validate_inside_epoch:
        return [checkpoint_best, checkpoint_epoch]
    else:
        # The last validation pass, whether at the end of an epoch or not
        checkpoint_last = _pl.callbacks.model_checkpoint.ModelCheckpoint(
            filename="checkpoint_last_{epoch:04d}_{step}", **kwargs
        )
        return [checkpoint_best, checkpoint_last, checkpoint_epoch]


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
    # Write
    for basename, config in (
        ("data", data_config),
        ("model", model_config),
        ("learning", learning_config),
    ):
        with open(_Path(outdir, f"config_{basename}.json"), "w") as fp:
            _json.dump(config, fp, indent=4)

    model = _LightningModule.init_from_config(model_config)
    # Add receptive field to data config:
    data_config["common"] = data_config.get("common", {})
    if "nx" in data_config["common"]:
        _warn(
            f"Overriding data nx={data_config['common']['nx']} with model requried {model.net.receptive_field}"
        )
    data_config["common"]["nx"] = model.net.receptive_field

    dataset_train = _init_dataset(data_config, _Split.TRAIN)
    dataset_validation = _init_dataset(data_config, _Split.VALIDATION)
    if dataset_train.sample_rate != dataset_validation.sample_rate:
        raise RuntimeError(
            "Train and validation data loaders have different data set sample rates: "
            f"{dataset_train.sample_rate}, {dataset_validation.sample_rate}"
        )
    model.net.sample_rate = dataset_train.sample_rate
    train_dataloader = _DataLoader(dataset_train, **learning_config["train_dataloader"])
    val_dataloader = _DataLoader(
        dataset_validation, **learning_config["val_dataloader"]
    )

    trainer = _pl.Trainer(
        callbacks=_create_callbacks(learning_config),
        default_root_dir=outdir,
        **learning_config["trainer"],
    )
    with _filter_warnings("ignore", category=_PossibleUserWarning):
        trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            **learning_config.get("trainer_fit_kwargs", {}),
        )
    # Go to best checkpoint
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    if best_checkpoint != "":
        model = _LightningModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            **_LightningModule.parse_config(model_config),
        )
    model.cpu()
    model.eval()
    if make_plots:
        _plot(
            model,
            dataset_validation,
            savefig=_Path(outdir, "comparison.png"),
            window_start=100_000,
            window_end=110_000,
            show=False,
        )
        _plot(model, dataset_validation, show=not no_show)
    # Export!
    model.net.export(outdir)
