"""
Test loading from a .nam file
"""

import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory
import pytest as _pytest

from nam.models import _from_nam
from nam.models.wavenet import WaveNet as _WaveNet


@_pytest.mark.parametrize(
    "factory,kwargs",
    (
        # A standard WaveNet
        (
            _WaveNet,  # i.e. .__init__()
            {
                "layers_configs": [
                    {
                        "condition_size": 1,
                        "input_size": 1,
                        "channels": 16,
                        "head_size": 8,
                        "kernel_size": 3,
                        "dilations": [
                            1,
                            2,
                            4,
                            8,
                            16,
                            32,
                            64,
                            128,
                            256,
                            512
                        ],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False
                    },
                    {
                        "condition_size": 1,
                        "input_size": 16,
                        "channels": 8,
                        "head_size": 1,
                        "kernel_size": 3,
                        "dilations": [
                            1,
                            2,
                            4,
                            8,
                            16,
                            32,
                            64,
                            128,
                            256,
                            512
                        ],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True
                    }
                ],
                "head_scale": 0.02
            }
        ),
    ),
)
def test_load_from_nam(factory, kwargs):
    """
    Assert that loading from a .nam file works by saving the model twice
    """
    model = factory(**kwargs)
    with _TemporaryDirectory() as tmpdir:
        model.export(_Path(tmpdir), basename="model")
        with open(_Path(tmpdir, "model.nam"), "r") as fp:
            nam_file_contents = _json.load(fp)
        model2 = _from_nam.init_from_nam(nam_file_contents)
        model2.export(_Path(tmpdir), basename="model2")
        with open(_Path(tmpdir, "model2.nam"), "r") as fp:
            nam_file_contents2 = _json.load(fp)

        # Metadata isn't preseved. At least creation time will be slightly different
        # Could improve this to preserve metadata on load
        nam_file_contents.pop("metadata")
        nam_file_contents2.pop("metadata")
        assert nam_file_contents == nam_file_contents2
    