# File: train.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nam.data import Split, init_dataset
from nam.models import Model

torch.manual_seed(0)


def timestamp() -> str:
    t = datetime.now()
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}-{t.second:02d}"


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
    with torch.no_grad():
        tx = len(ds.x) / 48_000
        print(f"Run (t={tx})")
        t0 = time()
        output = model(ds.x).flatten().cpu().numpy()
        t1 = time()
        print(f"Took {t1 - t0} ({tx / (t1 - t0):.2f}x)")

    plt.figure(figsize=(16, 5))
    plt.plot(ds.x[window_start:window_end], label="Input")
    plt.plot(output[window_start:window_end], label="Output")
    plt.plot(ds.y[window_start:window_end], label="Target")
    plt.title(f"NRMSE={_rms(torch.Tensor(output) - ds.y) / _rms(ds.y)}")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()


def main(args):
    outdir = ensure_outdir(args.outdir)
    # Read
    with open(args.data_config_path, "r") as fp:
        data_config = json.load(fp)
    with open(args.model_config_path, "r") as fp:
        model_config = json.load(fp)
    with open(args.learning_config_path, "r") as fp:
        learning_config = json.load(fp)
    # Write
    for basename, config in (
        ("data", data_config),
        ("model", model_config),
        ("learning", learning_config),
    ):
        with open(Path(outdir, f"config_{basename}.json"), "w") as fp:
            json.dump(config, fp, indent=4)

    model = Model.init_from_config(model_config)

    dataset_train = init_dataset(data_config, Split.TRAIN)
    dataset_validation = init_dataset(data_config, Split.VALIDATION)
    train_dataloader = DataLoader(dataset_train, **learning_config["train_dataloader"])
    val_dataloader = DataLoader(dataset_validation, **learning_config["val_dataloader"])

    # ckpt_path = Path(outdir, "checkpoints")
    # ckpt_path.mkdir()
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                filename="{epoch}_{val_loss:.6f}",
                save_top_k=3,
                monitor="val_loss",
                every_n_epochs=1,
            ),
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                filename="checkpoint_last_{epoch:04d}", every_n_epochs=1
            ),
        ],
        default_root_dir=outdir,
        **learning_config["trainer"],
    )
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        **learning_config.get("trainer_fit_kwargs", {}),
    )
    # Go to best checkpoint
    model = Model.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, **Model.parse_config(model_config)
    )
    model.eval()
    plot(
        model,
        dataset_validation,
        savefig=Path(outdir, "comparison.png"),
        window_start=100_000,
        window_end=110_000,
        show=False,
    )
    plot(model, dataset_validation)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_config_path", type=str)
    parser.add_argument("model_config_path", type=str)
    parser.add_argument("learning_config_path", type=str)
    parser.add_argument("outdir")
    main(parser.parse_args())
