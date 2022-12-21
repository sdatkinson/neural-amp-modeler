# File: gui.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Functions used by the GUI trainer.
"""

import hashlib
from time import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..data import REQUIRED_RATE, Split, init_dataset, wav_to_np
from ..models import Model
from ._version import Version


def _detect_input_version(input_path) -> Version:
    """
    Check to see if the input matches any of the known inputs
    """
    md5 = hashlib.md5()
    buffer_size = 65536
    with open(input_path, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            md5.update(data)
    file_hash = md5.hexdigest()

    version = {
        "4d54a958861bf720ec4637f43d44a7ef": Version(1, 0, 0),
        "7c3b6119c74465f79d96c761a0e27370": Version(1, 1, 1),
    }.get(file_hash)
    if version is None:
        raise RuntimeError(
            f"Provided input file {input_path} does not match any known standard input "
            "files."
        )
    return version


def _calibrate_delay_v1(input_path, output_path) -> int:
    safety_factor = 4
    # Locations of blips in v1 signal file:
    i1, i2 = 12_000, 36_000
    j1_start_looking = i1 - 1_000
    j2_start_looking = i2 - 1_000

    y = wav_to_np(output_path)[:48_000]

    background_level = np.max(np.abs(y[:6_000]))
    trigger_threshold = max(background_level + 0.01, 1.01 * background_level)
    j1 = np.where(np.abs(y[j1_start_looking:j2_start_looking]) > trigger_threshold)[0][
        0
    ]
    j2 = np.where(np.abs(y[j2_start_looking:]) > trigger_threshold)[0][0]

    delay_1 = (j1 + j1_start_looking) - i1
    delay_2 = (j2 + j2_start_looking) - i2
    print(f"Delays: {delay_1}, {delay_2}")
    delay = int(np.min([delay_1, delay_2])) - safety_factor
    print(f"Final delay is {delay}")
    return delay


def _plot_delay_v1(delay: int, input_path: str, output_path: str):
    print("Plotting the delay for manual inspection...")
    x = wav_to_np(input_path)[:48_000]
    y = wav_to_np(output_path)[:48_000]
    i = np.where(np.abs(x) > 0.1)[0][0]  # In case resampled poorly
    di = 20
    plt.figure()
    # plt.plot(x[i - di : i + di], ".-", label="Input")
    plt.plot(
        np.arange(-di, di),
        y[i - di + delay : i + di + delay],
        ".-",
        label="Output",
    )
    plt.axvline(x=0, linestyle="--", color="C1")
    plt.legend()
    plt.show()  # This doesn't freeze the notebook


def _calibrate_delay(
    delay: Optional[int], input_version: Version, input_path: str, output_path: str,
) -> int:
    if input_version.major == 1:
        calibrate, plot = _calibrate_delay_v1, _plot_delay_v1
    else:
        raise NotImplementedError(
            f"Input calibration not implemented for input version {input_version}"
        )
    if delay is not None:
        print(f"Delay is specified as {delay}")
    else:
        print("Delay wasn't provided; attempting to calibrate automatically...")
        delay = calibrate(input_path, output_path)
    plot(delay, input_path, output_path)
    return delay


def _get_configs(
    input_basename: str,
    output_basename: str,
    delay: int,
    epochs: int,
    stage_1_channels: int,
    stage_2_channels: int,
    head_scale: float,
    lr: float,
    lr_decay: float,
):
    val_seconds = 9
    train_val_split = -val_seconds * REQUIRED_RATE
    data_config = {
        "train": {"ny": 8192, "stop": train_val_split},
        "validation": {"ny": None, "start": train_val_split},
        "common": {
            "x_path": input_basename,
            "y_path": output_basename,
            "delay": delay,
        },
    }
    model_config = {
        "net": {
            "name": "WaveNet",
            # This should do decently. If you really want a nice model, try turning up
            # "channels" in the first block and "input_size" in the second from 12 to 16.
            "config": {
                "layers_configs": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "head_size": stage_2_channels,
                        "channels": stage_1_channels,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False,
                    },
                    {
                        "input_size": stage_1_channels,
                        "condition_size": 1,
                        "head_size": 1,
                        "channels": stage_2_channels,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True,
                    },
                ],
                "head_scale": head_scale,
            },
        },
        "loss": {"val_loss": "esr"},
        "optimizer": {"lr": lr},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 1.0 - lr_decay}},
    }
    if torch.cuda.is_available():
        device_config = {"accelerator": "gpu", "devices": 1}
    else:
        print("WARNING: No GPU was found. Training will be very slow!")
        device_config = {}
    learning_config = {
        "train_dataloader": {
            "batch_size": 16,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {},
        "trainer": {"max_epochs": epochs, **device_config},
    }
    return data_config, model_config, learning_config


def _esr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (
        torch.mean(torch.square(pred - target)).item()
        / torch.mean(torch.square(target)).item()
    )


def _plot(
    model, ds, window_start: Optional[int] = None, window_end: Optional[int] = None
):
    print("Plotting a comparison of your model with the target output...")
    with torch.no_grad():
        tx = len(ds.x) / 48_000
        print(f"Run (t={tx:.2f} sec)")
        t0 = time()
        output = model(ds.x).flatten().cpu().numpy()
        t1 = time()
        print(f"Took {t1 - t0:.2f} sec ({tx / (t1 - t0):.2f}x)")

    esr = _esr(torch.Tensor(output), ds.y)
    # Trying my best to put numbers to it...
    if esr < 0.01:
        esr_comment = "Great!"
    elif esr < 0.035:
        esr_comment = "Not bad!"
    elif esr < 0.1:
        esr_comment = "...This *might* sound ok!"
    elif esr < 0.3:
        esr_comment = "...This probably won't sound great :("
    else:
        esr_comment = "...Something seems to have gone wrong."
    print(f"Error-signal ratio = {esr:.3f}")
    print(esr_comment)

    plt.figure(figsize=(16, 5))
    plt.plot(output[window_start:window_end], label="Prediction")
    plt.plot(ds.y[window_start:window_end], linestyle="--", label="Target")
    plt.title(f"ESR={esr:.3f}")
    plt.legend()
    plt.show()


def train(
    input_path: str,
    output_path: str,
    train_path: str,
    input_version: Optional[Version] = None,
    epochs=100,
    delay=None,
    stage_1_channels=16,
    stage_2_channels=8,
    head_scale: float = 0.02,
    lr=0.004,
    lr_decay=0.007,
    seed: Optional[int] = 0,
):
    if seed is not None:
        torch.manual_seed(seed)

    if delay is None:
        if input_version is None:
            input_version = _detect_input_version(input_path)
    delay = _calibrate_delay(delay, input_version, input_path, output_path)

    data_config, model_config, learning_config = _get_configs(
        input_path,
        output_path,
        delay,
        epochs,
        stage_1_channels,
        stage_2_channels,
        head_scale,
        lr,
        lr_decay,
    )

    print("Starting training. Let's rock!")
    model = Model.init_from_config(model_config)
    data_config["common"]["nx"] = model.net.receptive_field
    dataset_train = init_dataset(data_config, Split.TRAIN)
    dataset_validation = init_dataset(data_config, Split.VALIDATION)
    train_dataloader = DataLoader(dataset_train, **learning_config["train_dataloader"])
    val_dataloader = DataLoader(dataset_validation, **learning_config["val_dataloader"])

    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                filename="checkpoint_best_{epoch:04d}_{step}_{ESR:.4f}_{MSE:.3e}",
                save_top_k=3,
                monitor="val_loss",
                every_n_epochs=1,
            ),
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                filename="checkpoint_last_{epoch:04d}_{step}", every_n_epochs=1
            ),
        ],
        default_root_dir=train_path,
        **learning_config["trainer"],
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # Go to best checkpoint
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    if best_checkpoint != "":
        model = Model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            **Model.parse_config(model_config),
        )
    model.eval()

    _plot(
        model,
        dataset_validation,
        window_start=100_000,  # Start of the plotting window, in samples
        window_end=101_000,  # End of the plotting window, in samples
    )
    return model
