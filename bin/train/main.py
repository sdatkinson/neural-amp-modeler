# File: train.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)


# Hack to recover graceful shutdowns in Windows.
# This has to happen ASAP
# See:
# https://github.com/sdatkinson/neural-amp-modeler/issues/105
# https://stackoverflow.com/a/44822794
def _ensure_graceful_shutdowns():
    import os

    if os.name == "nt":  # OS is Windows
        os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"


_ensure_graceful_shutdowns()

import json
from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import Optional, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import torch
from torch.utils.data import DataLoader

from nam.data import ConcatDataset, ParametricDataset, Split, init_dataset
from nam.models import Model
from nam.util import filter_warnings, timestamp

torch.manual_seed(0)


def ensure_outdir(outdir: str) -> Path:
    outdir = Path(outdir, timestamp())
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def _rms(x: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(x, np.ndarray):
        return np.sqrt(np.mean(np.square(x)))
    elif isinstance(x, torch.Tensor):
        return torch.sqrt(torch.mean(torch.square(x))).item()
    else:
        raise TypeError(type(x))


def plot(
    model,
    ds,
    savefig=None,
    show=True,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
):
    if isinstance(ds, ConcatDataset):

        def extend_savefig(i, savefig):
            if savefig is None:
                return None
            savefig = Path(savefig)
            extension = savefig.name.split(".")[-1]
            stem = savefig.name[: -len(extension) - 1]
            return Path(savefig.parent, f"{stem}_{i}.{extension}")

        for i, ds_i in enumerate(ds.datasets):
            plot(
                model,
                ds_i,
                savefig=extend_savefig(i, savefig),
                show=show and i == len(ds.datasets) - 1,
                window_start=window_start,
                window_end=window_end,
            )
        return
    with torch.no_grad():
        tx = len(ds.x) / 48_000
        print(f"Run (t={tx:.2f})")
        t0 = time()
        args = (ds.vals, ds.x) if isinstance(ds, ParametricDataset) else (ds.x,)
        output = model(*args).flatten().cpu().numpy()
        t1 = time()
        try:
            rt = f"{tx / (t1 - t0):.2f}"
        except ZeroDivisionError as e:
            rt = "???"
        print(f"Took {t1 - t0:.2f} ({rt}x)")

    plt.figure(figsize=(16, 5))
    # plt.plot(ds.x[window_start:window_end], label="Input")
    plt.plot(output[window_start:window_end], label="Prediction")
    plt.plot(ds.y[window_start:window_end], linestyle="--", label="Target")
    # plt.plot(
    #     ds.y[window_start:window_end] - output[window_start:window_end], label="Error"
    # )
    nrmse = _rms(torch.Tensor(output) - ds.y) / _rms(ds.y)
    esr = nrmse**2
    plt.title(f"ESR={esr:.3f}")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()


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

    checkpoint_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename="{epoch:04d}_{step}_{ESR:.3e}_{MSE:.3e}",
        save_top_k=3,
        monitor="val_loss",
        **kwargs,
    )

    # return [checkpoint_best, checkpoint_last]
    # The last epoch that was finished.
    checkpoint_epoch = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename="checkpoint_epoch_{epoch:04d}", every_n_epochs=1
    )
    if not validate_inside_epoch:
        return [checkpoint_best, checkpoint_epoch]
    else:
        # The last validation pass, whether at the end of an epoch or not
        checkpoint_last = pl.callbacks.model_checkpoint.ModelCheckpoint(
            filename="checkpoint_last_{epoch:04d}_{step}", **kwargs
        )
        return [checkpoint_best, checkpoint_last, checkpoint_epoch]


def main(args):
    outdir = ensure_outdir(args.outdir)
    # Read
    with open(args.data_config_path, "r") as fp:
        data_config = json.load(fp)
    with open(args.model_config_path, "r") as fp:
        model_config = json.load(fp)
    with open(args.learning_config_path, "r") as fp:
        learning_config = json.load(fp)
    main_inner(data_config, model_config, learning_config, outdir, args.no_show)


def main_inner(
    data_config, model_config, learning_config, outdir, no_show, make_plots=True
):
    # Write
    for basename, config in (
        ("data", data_config),
        ("model", model_config),
        ("learning", learning_config),
    ):
        with open(Path(outdir, f"config_{basename}.json"), "w") as fp:
            json.dump(config, fp, indent=4)

    model = Model.init_from_config(model_config)
    # Add receptive field to data config:
    data_config["common"] = data_config.get("common", {})
    if "nx" in data_config["common"]:
        warn(
            f"Overriding data nx={data_config['common']['nx']} with model requried {model.net.receptive_field}"
        )
    data_config["common"]["nx"] = model.net.receptive_field

    dataset_train = init_dataset(data_config, Split.TRAIN)
    dataset_validation = init_dataset(data_config, Split.VALIDATION)
    if dataset_train.sample_rate != dataset_validation.sample_rate:
        raise RuntimeError(
            "Train and validation data loaders have different data set sample rates: "
            f"{dataset_train.sample_rate}, {dataset_validation.sample_rate}"
        )
    train_dataloader = DataLoader(dataset_train, **learning_config["train_dataloader"])
    val_dataloader = DataLoader(dataset_validation, **learning_config["val_dataloader"])

    trainer = pl.Trainer(
        callbacks=_create_callbacks(learning_config),
        default_root_dir=outdir,
        **learning_config["trainer"],
    )
    with filter_warnings("ignore", category=PossibleUserWarning):
        trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            **learning_config.get("trainer_fit_kwargs", {}),
        )
    # Go to best checkpoint
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    if best_checkpoint != "":
        model = Model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            **Model.parse_config(model_config),
        )
    model.cpu()
    model.eval()
    model.net.sample_rate = train_dataloader.dataset.sample_rate
    if make_plots:
        plot(
            model,
            dataset_validation,
            savefig=Path(outdir, "comparison.png"),
            window_start=100_000,
            window_end=110_000,
            show=False,
        )
        plot(model, dataset_validation, show=not no_show)
    # Would like to, but this doesn't work for all cases.
    # If you're making snapshot models, you may find this convenient to uncomment :)
    # model.net.export(outdir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_config_path", type=str)
    parser.add_argument("model_config_path", type=str)
    parser.add_argument("learning_config_path", type=str)
    parser.add_argument("outdir")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")
    main(parser.parse_args())
