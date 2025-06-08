"""
Test loading from a .nam file
"""

import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory
from typing import Callable as _Callable, Optional as _Optional
import pytest as _pytest

from nam.models import _from_nam
from nam.models.linear import Linear as _Linear
from nam.models.recurrent import LSTM as _LSTM
from nam.models.wavenet import WaveNet as _WaveNet


def _default_comparison(expected, actual):
    assert actual == expected


class _MockLSTM(_LSTM):
    """
    Just a shorter burn-in
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_initial_state_burn_in = 1

    def _get_export_architecture(self):
        # Ope, not "_MockLSTM"!
        return "LSTM"


def _compare_lstm_configs(
    expected_nam_file_contents: dict, actual_nam_file_contents: dict
):
    """
    We shouldn't expect that the inital hidden and cell states will be exactly
    the same because I run a burn-in when exporting.
    """

    def assert_weights(expected_weights, actual_weights):
        """
        Skip weights associated with the initial hidden and cell states
        """

        def get_subarrays():
            input_size = actual_nam_file_contents["config"]["input_size"]
            hidden_size = actual_nam_file_contents["config"]["hidden_size"]
            num_layers = actual_nam_file_contents["config"]["num_layers"]
            weight_index = 0
            for i in range(num_layers):
                # Weights (do all at once)
                # i,h -> ifgo
                n_w = (4 * hidden_size) * (input_size + hidden_size)
                yield weight_index, weight_index + n_w
                weight_index += n_w
                # bias (there's only one, merged input/hidden)
                n_b = 4 * hidden_size
                yield weight_index, weight_index + n_b
                weight_index += n_b

                # Skip initial hidden and cell states--these will probably be differnt
                # yield hidden_state
                # yield cell_state
                weight_index += 2 * hidden_size

                # For next layers, hidden state from previous cell is input
                input_size = hidden_size

            # Head weights: (dh x 1 + bias)
            n_head = hidden_size + 1
            yield weight_index, weight_index + n_head
            weight_index += n_head
            assert weight_index == len(expected_weights)
            assert weight_index == len(actual_weights)

        for i, j in get_subarrays():
            print(f"i: {i}, j: {j}")
            assert expected_weights[i:j] == actual_weights[i:j]

    assert set(expected_nam_file_contents.keys()) == set(
        actual_nam_file_contents.keys()
    )
    for key, actual_value in actual_nam_file_contents.items():
        expected_value = expected_nam_file_contents[key]
        if key == "weights":
            continue  # Do this last
        else:
            assert actual_value == expected_value

    assert "weights" in actual_nam_file_contents
    assert_weights(
        expected_nam_file_contents["weights"], actual_nam_file_contents["weights"]
    )


@_pytest.mark.parametrize(
    "factory,kwargs,comparison",
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
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False,
                    },
                    {
                        "condition_size": 1,
                        "input_size": 16,
                        "channels": 8,
                        "head_size": 1,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True,
                    },
                ],
                "head_scale": 0.02,
            },
            _default_comparison,
        ),
        # LSTM
        (
            _MockLSTM,
            {"input_size": 1, "hidden_size": 3, "num_layers": 2},
            _compare_lstm_configs,
        ),
        # Linear (an IR)
        (_Linear, {"receptive_field": 2, "bias": False}, _default_comparison),
    ),
)
def test_load_from_nam(
    factory, kwargs, comparison: _Optional[_Callable[[dict, dict], None]]
):
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
        comparison(nam_file_contents, nam_file_contents2)  # type: ignore assert
