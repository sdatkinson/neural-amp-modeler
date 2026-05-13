import os
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import pytest
from pathlib import Path
# NAM imports: use the real LightningModule + the custom checkpoint class
from nam.train.lightning_module import LightningModule, LossConfig
from nam.train.core import _ModelCheckpoint  # private but fine for a repo test


class _MockNamNet(nn.Module):
    """
    Minimal NAM-like net:
      - has sample_rate (used by on_save_checkpoint/on_load_checkpoint)
      - forward accepts pad_start kwarg (LightningModule calls net(..., pad_start=False))
      - at least one trainable param so optimizer is valid
    """
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x, pad_start: bool = False):
        return x * self.gain
    
    def export(self, outdir, basename, user_metadata=None, other_metadata=None):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        nam_path = outdir / f"{basename}.nam"
        # Write a tiny, valid-ish artifact so the test can assert existence
        # (content doesn't matter for this test)
        nam_path.write_bytes(b"NA M")


def _fast_loader(n_steps=300, batch_size=1, seq_len=1024):
    # Very small tensors so each step is *extremely* fast
    x = torch.randn(n_steps * batch_size, seq_len)
    y = torch.randn(n_steps * batch_size, seq_len)
    ds = TensorDataset(x, y)  # LightningModule expects (args..., target) -> here (x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _nam_model():
    net = _MockNamNet(sample_rate=48000)
    loss_cfg = LossConfig(mse_weight=1.0, mrstft_weight=None)  # keep deps light
    return LightningModule(net=net, loss_config=loss_cfg)


def _trainer_with_checkpoint(tmpdir, *, every_n_train_steps=1, filename="constant"):
    # Use NAM's custom checkpoint class so both .ckpt and .nam writes are exercised
    ckpt = _ModelCheckpoint(
        dirpath=tmpdir,
        filename=filename,
        save_top_k=-1,
        every_n_train_steps=every_n_train_steps,
        save_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=300,
        logger=False,
        enable_checkpointing=True,
        callbacks=[ckpt],
        enable_model_summary=False,
        accelerator="cpu",
        precision="32-true",
        log_every_n_steps=50,
        deterministic=False,
    )
    return trainer


def test_tight_loop_save_every_step_no_crash():
    """
    Tight loop: save a checkpoint on *every* train step with a constant filename.
    Before the fix, this can intermittently crash on some systems.
    After the fix (retry-on-transient-I/O), it should complete and produce files.
    """
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer_with_checkpoint(tmp, every_n_train_steps=1, filename="constant")
        trainer.fit(_nam_model(), train_dataloaders=_fast_loader())

        # Sanity: we should have at least one .ckpt and the companion .nam
        files = os.listdir(tmp)
        assert any(f.endswith(".ckpt") for f in files), f"No .ckpt found in {files}"
        assert any(f.endswith(".nam") for f in files), f"No .nam found in {files}"


def test_tight_loop_transient_write_is_retried(monkeypatch):
    """
    Deterministic repro of transient I/O: simulate intermittent failure in torch.save.
    The retry wrapper added to _ModelCheckpoint._save_checkpoint should make this pass.
    """
    calls = {"n": 0}
    real_save = torch.save

    def flaky_save(*args, **kwargs):
        calls["n"] += 1
        # Fail 2 of every 3 calls to mimic a busy/locked FS; succeed on the 3rd
        if calls["n"] % 3 != 0:
            raise PermissionError("Simulated transient write failure")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", flaky_save, raising=True)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = _trainer_with_checkpoint(tmp, every_n_train_steps=1, filename="constant")
        trainer.fit(_nam_model(), train_dataloaders=_fast_loader())

        files = os.listdir(tmp)
        assert any(f.endswith(".ckpt") for f in files)
        assert any(f.endswith(".nam") for f in files)
