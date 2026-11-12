import json

import pytorch_lightning as _pl

from nam.train import full as _full

from .test_full_packed import _data_config, _learning_config, _model_config
from .test_full_packed import _write_wav_pair


class _RecordingCallback(_pl.Callback):
    """
    Records its construction arguments and whether training started, so tests can
    check that callbacks from the learning config are built and attached.
    """

    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.train_started = False
        _RecordingCallback.instances.append(self)

    def on_train_start(self, trainer, pl_module):
        self.train_started = True


_RECORDING_CALLBACK_NAME = f"{__name__}._RecordingCallback"


def test_init_config_callbacks_empty():
    assert _full._init_config_callbacks([]) == []


def test_init_config_callbacks_args_and_kwargs():
    callbacks = _full._init_config_callbacks(
        [
            {"name": _RECORDING_CALLBACK_NAME, "args": [1, 2], "kwargs": {"a": 3}},
            {"name": _RECORDING_CALLBACK_NAME},
        ]
    )
    assert len(callbacks) == 2
    assert all(isinstance(c, _RecordingCallback) for c in callbacks)
    assert callbacks[0].args == (1, 2)
    assert callbacks[0].kwargs == {"a": 3}
    assert callbacks[1].args == ()
    assert callbacks[1].kwargs == {}


def test_full_main_attaches_config_callbacks(tmp_path):
    _RecordingCallback.instances.clear()
    x_path, y_path = _write_wav_pair(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    learning_config = _learning_config()
    callback_configs = [{"name": _RECORDING_CALLBACK_NAME, "kwargs": {"tag": "x"}}]
    learning_config["trainer"]["callbacks"] = callback_configs

    _full.main(
        _data_config(x_path, y_path),
        _model_config(),
        learning_config,
        outdir,
        no_show=True,
        make_plots=False,
    )

    assert len(_RecordingCallback.instances) == 1
    callback = _RecordingCallback.instances[0]
    assert callback.kwargs == {"tag": "x"}
    assert callback.train_started
    # The caller's config isn't modified, and the saved copy keeps the callbacks
    assert learning_config["trainer"]["callbacks"] == callback_configs
    with open(outdir / "config_learning.json", "r") as fp:
        assert json.load(fp)["trainer"]["callbacks"] == callback_configs
