# File: colab.py
# Created Date: Sunday December 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Hide the mess in Colab to make things look pretty for users.
"""

from pathlib import Path
from time import time
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nam.data import REQUIRED_RATE, Split, init_dataset, wav_to_np
from nam.models import Model

_INPUT_BASENAME = "v1.wav"
_OUTPUT_BASENAME = "output.wav"


def _check_for_files():
    print("Checking that we have all of the required audio files...")
    if not Path(_INPUT_BASENAME).exists():
        raise FileNotFoundError(
            f"Didn't find NAM's input audio file. Please upload {_INPUT_BASENAME}"
        )
    if not Path(_OUTPUT_BASENAME).exists():
        raise FileNotFoundError(
            f"Didn't find your reamped output audio file. Please upload {_OUTPUT_BASENAME}."
        )


def _calibrate_delay_v1() -> int:
    safety_factor = 4
    # Locations of blips in v1 signal file:
    i1, i2 = 12_000, 36_000
    j1_start_looking = i1 - 1_000
    j2_start_looking = i2 - 1_000

    y = wav_to_np(_OUTPUT_BASENAME)[:48_000]

    background_level = np.max(np.abs(y[:6_000]))
    trigger_threshold = background_level + 0.01
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


def _calibrate_delay() -> int:
    print("Delay wasn't provided; attempting to calibrate automatically...")
    return _calibrate_delay_v1()


def _get_configs(delay: int, epochs: int, stage_1_channels, stage_2_channels):
    val_seconds = 9
    train_val_split = -val_seconds * REQUIRED_RATE
    data_config = {
        "train": {"ny": 8192, "stop": train_val_split},
        "validation": {"ny": None, "start": train_val_split},
        "common": {
            "x_path": _INPUT_BASENAME,
            "y_path": _OUTPUT_BASENAME,
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
                ]
            },
        },
        "loss": {"val_loss": "esr"},
        "optimizer": {"lr": 0.003},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
    }
    learning_config = {
        "train_dataloader": {
            "batch_size": 16,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {},
        "trainer": {"accelerator": "gpu", "devices": 1, "max_epochs": epochs},
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


def _get_valid_export_directory():
    def get_path(version):
        return Path("exported_models", f"version_{version}")

    version = 1
    while get_path(version).exists():
        version += 1
    return get_path(version)


def run(epochs=100, delay=None, stage_1_channels=16, stage_2_channels=8):
    _check_for_files()
    delay = _calibrate_delay() if delay is None else delay
    data_config, model_config, learning_config = _get_configs(
        delay, epochs, stage_1_channels, stage_2_channels
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

    print("Exporting your model...")
    model_export_outdir = _get_valid_export_directory()
    model_export_outdir.mkdir(parents=True, exist_ok=False)
    model.net.export(model_export_outdir)
    print(f"Model exported to {model_export_outdir}. Enjoy!")
