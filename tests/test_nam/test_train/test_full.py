# File: test_full.py
# Created Date: Saturday May 3rd 2025
# Author: Elias Gomes

from unittest.mock import MagicMock as _MagicMock

import pytest as _pytest

from nam.train.full import _EpochCountColumn, _EpochProgressBar


class TestEpochCountColumn:
    @staticmethod
    def _make_task(completed, total):
        task = _MagicMock()
        task.completed = completed
        task.total = total
        return task

    def test_mid_epoch(self):
        col = _EpochCountColumn()
        assert col.render(self._make_task(3.5, 100)).plain == "3/99"

    def test_epoch_end(self):
        col = _EpochCountColumn()
        assert col.render(self._make_task(4, 100)).plain == "4/99"

    def test_training_complete(self):
        col = _EpochCountColumn()
        assert col.render(self._make_task(100, 100)).plain == "99/99"


class TestEpochProgressBar:
    def test_init(self):
        bar = _EpochProgressBar()
        assert bar._epoch_progress is None
        assert bar._epoch_task_id is None
        assert bar._leave is False

    @staticmethod
    def _make_trainer_mock(**pbar_metrics):
        trainer = _MagicMock()
        trainer.progress_bar_metrics = pbar_metrics
        trainer.state.fn = None
        trainer.loggers = [_MagicMock(version=0)]
        return trainer

    def test_get_metrics_removes_v_num(self):
        bar = _EpochProgressBar()
        trainer = self._make_trainer_mock(val_loss=0.1)
        metrics = bar.get_metrics(trainer, _MagicMock())
        assert "v_num" not in metrics

    def test_get_metrics_preserves_other_keys(self):
        bar = _EpochProgressBar()
        trainer = self._make_trainer_mock(val_loss=0.5, ESR=0.01)
        metrics = bar.get_metrics(trainer, _MagicMock())
        assert "val_loss" in metrics
        assert "ESR" in metrics

    def test_train_description_omits_total(self):
        bar = _EpochProgressBar()
        assert bar._get_train_description(5) == "Epoch 5"

    def test_on_train_start_creates_epoch_progress(self):
        bar = _EpochProgressBar()
        trainer = _MagicMock()
        trainer.max_epochs = 100
        pl_module = _MagicMock()

        bar._init_progress(trainer)
        bar.on_train_start(trainer, pl_module)

        assert bar._epoch_progress is not None
        assert bar._epoch_task_id is not None

    def test_on_train_start_skips_when_max_epochs_is_none(self):
        bar = _EpochProgressBar()
        trainer = _MagicMock()
        trainer.max_epochs = None
        pl_module = _MagicMock()

        bar._init_progress(trainer)
        bar.on_train_start(trainer, pl_module)

        assert bar._epoch_progress is None
        assert bar._epoch_task_id is None

    def test_on_train_epoch_end_updates_completed(self):
        bar = _EpochProgressBar()
        trainer = _MagicMock()
        trainer.max_epochs = 10
        trainer.current_epoch = 3
        pl_module = _MagicMock()

        bar._init_progress(trainer)
        bar.on_train_start(trainer, pl_module)
        bar.on_train_epoch_end(trainer, pl_module)

        task = bar._epoch_progress.tasks[bar._epoch_task_id]
        assert task.completed == 4

    def test_stop_epoch_progress_transient(self):
        bar = _EpochProgressBar()
        trainer = _MagicMock()
        trainer.max_epochs = 10
        pl_module = _MagicMock()

        bar._init_progress(trainer)
        bar.on_train_start(trainer, pl_module)
        assert bar._epoch_progress.live.is_started

        bar._stop_epoch_progress()
        assert not bar._epoch_progress.live.is_started
        assert bar._epoch_progress.live.transient

    def test_stop_epoch_progress_leave(self):
        bar = _EpochProgressBar()
        trainer = _MagicMock()
        trainer.max_epochs = 10
        pl_module = _MagicMock()

        bar._init_progress(trainer)
        bar.on_train_start(trainer, pl_module)
        bar._stop_epoch_progress(leave=True)
        assert not bar._epoch_progress.live.transient

    def test_stop_epoch_progress_noop_when_none(self):
        bar = _EpochProgressBar()
        bar._stop_epoch_progress()

    def test_on_train_end_leaves_epoch_bar(self):
        bar = _EpochProgressBar()
        trainer = _MagicMock()
        trainer.max_epochs = 10
        pl_module = _MagicMock()

        bar._init_progress(trainer)
        bar.on_train_start(trainer, pl_module)
        bar.on_train_end(trainer, pl_module)
        assert not bar._epoch_progress.live.is_started
        assert not bar._epoch_progress.live.transient
        assert bar._progress_stopped

    def test_on_exception_stops_all(self):
        bar = _EpochProgressBar()
        trainer = _MagicMock()
        trainer.max_epochs = 10
        pl_module = _MagicMock()

        bar._init_progress(trainer)
        bar.on_train_start(trainer, pl_module)
        bar.on_exception(trainer, pl_module, RuntimeError("test"))
        assert not bar._epoch_progress.live.is_started
        assert bar._epoch_progress.live.transient
        assert bar._progress_stopped


if __name__ == "__main__":
    _pytest.main()
