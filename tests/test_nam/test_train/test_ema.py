"""Optional EMA callback wiring for nam-full."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EMAWeightAveraging

from nam.train.ema import ema_callback_from_learning_config


def test_ema_callback_absent_or_disabled_returns_none():
    assert ema_callback_from_learning_config({}) is None
    assert ema_callback_from_learning_config({"ema": {"enabled": False}}) is None


def test_ema_callback_enabled_returns_averaging_instance():
    cb = ema_callback_from_learning_config(
        {"ema": {"enabled": True, "decay": 0.99, "use_buffers": False}}
    )
    assert isinstance(cb, EMAWeightAveraging)


def test_full_main_callback_list_includes_ema_when_enabled():
    from nam.train.full import _create_callbacks

    learning_config = {
        "trainer": {"check_val_every_n_epoch": 1},
        "ema": {"enabled": True, "decay": 0.999},
    }
    callbacks = _create_callbacks(learning_config)
    ema_cb = ema_callback_from_learning_config(learning_config)
    assert ema_cb is not None
    merged = [*callbacks, ema_cb]
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=0,
        limit_val_batches=0,
        enable_progress_bar=False,
        callbacks=merged,
    )
    assert trainer is not None
    assert any(isinstance(c, EMAWeightAveraging) for c in trainer.callbacks)
