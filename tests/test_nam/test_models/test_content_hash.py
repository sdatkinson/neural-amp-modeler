"""
Tests for `metadata.weights_hash`
"""

import hashlib
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from nam import data as _data
from nam.models import content_hash as _ch
from nam.models import linear as _linear
from nam.models import sequential as _sequential
from nam.models.wavenet import _packed_wavenet as _pw

from .test_packed_wavenet import _packed_config


_VECTORS_PATH = (
    Path(__file__).parents[2] / "resources" / "content_hash_test_vectors.json"
)


def _reference(*leaves) -> str:
    """
    Independent reference: tag, then each leaf's u64le count and f32le weights.
    """
    h = hashlib.sha256(b"nam-content-v1\0")
    for leaf in leaves:
        h.update(struct.pack("<Q", len(leaf)) + struct.pack(f"<{len(leaf)}f", *leaf))
    return f"nam-content-v1:{h.hexdigest()}"


def _vectors():
    with open(_VECTORS_PATH, "r") as fp:
        return json.load(fp)["vectors"]


@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: v["name"])
def test_vectors(vector):
    if vector["weights_hash"] is None:
        with pytest.raises(_ch.UnhashableModelError):
            _ch.weights_hash(vector["model"])
    else:
        assert _ch.weights_hash(vector["model"]) == vector["weights_hash"]


def test_vectors_catch_a_wrong_hash():
    vectors = {v["name"]: v for v in _vectors()}
    assert (
        _ch.weights_hash(vectors["weight_order_matters"]["model"])
        != vectors["wavenet"]["weights_hash"]
    )


def test_plain_model_hashes_little_endian_float32_weights():
    d = {"architecture": "WaveNet", "config": {}, "weights": [0.5, -1.25, 3.0]}
    assert _ch.weights_hash(d) == _reference([0.5, -1.25, 3.0])


def test_hash_ignores_metadata_and_config():
    a = {"architecture": "WaveNet", "config": {"a": 1}, "weights": [1.0, 2.0]}
    b = {
        "architecture": "LSTM",
        "config": {"b": 2},
        "metadata": {"name": "x", "loudness": -20.0},
        "weights": [1.0, 2.0],
    }
    assert _ch.weights_hash(a) == _ch.weights_hash(b)


def test_hash_changes_with_weights():
    a = {"architecture": "WaveNet", "config": {}, "weights": [1.0, 2.0]}
    b = {"architecture": "WaveNet", "config": {}, "weights": [1.0, 2.5]}
    c = {"architecture": "WaveNet", "config": {}, "weights": [2.0, 1.0]}
    assert len({_ch.weights_hash(x) for x in (a, b, c)}) == 3


def test_slimmable_container_hashes_submodels_small_to_large():
    small = {"architecture": "WaveNet", "config": {}, "weights": [1.0]}
    big = {"architecture": "WaveNet", "config": {}, "weights": [2.0, 3.0]}
    container = {
        "architecture": "SlimmableContainer",
        "config": {
            # Deliberately listed large-first; the hash must sort by max_value.
            "submodels": [
                {"max_value": 1.0, "model": big},
                {"max_value": 0.5, "model": small},
            ]
        },
        "weights": [],
    }
    assert _ch.weights_hash(container) == _reference([1.0], [2.0, 3.0])


def test_sequential_hashes_children_in_processing_order():
    first = {"architecture": "Linear", "config": {}, "weights": [1.0, 2.0]}
    second = {"architecture": "Linear", "config": {}, "weights": [3.0]}
    seq = {
        "architecture": "Sequential",
        "config": {"models": [first, second]},
        "weights": [],
    }
    assert _ch.weights_hash(seq) == _reference([1.0, 2.0], [3.0])


def test_export_stamps_weights_hash():
    model = _linear.Linear(receptive_field=2, sample_rate=48_000)
    with TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        with open(Path(tmpdir, "model.nam"), "r") as fp:
            model_dict = json.load(fp)
    stamped = model_dict["metadata"]["weights_hash"]
    assert isinstance(stamped, str) and stamped.startswith("nam-content-v1:")
    assert "weights_hash" not in model_dict
    assert "content_hash" not in model_dict["metadata"]
    # The stamp matches a recomputation from the file as written.
    assert stamped == _ch.weights_hash(model_dict)


def test_export_stamps_hash_after_post_hooks_change_weights(tmp_path):
    """
    Post hooks (e.g. the dataset output-scale handshake) rescale weights on the
    way out; the hash must describe the weights actually written.
    """
    model = _pw.PackedWaveNet.init_from_config(
        {**_packed_config(), "sample_rate": 48_000}
    )
    plain = model.export_container(tmp_path)
    model.export_model_dict_post_hooks.append(_data.Dataset._ScaleOutputHook(scale=2.0))
    scaled = model.export_container(tmp_path)
    with open(tmp_path / "model.nam", "r") as fp:
        from_disk = json.load(fp)

    assert scaled["metadata"]["weights_hash"] == _ch.weights_hash(from_disk)
    assert scaled["metadata"]["weights_hash"] != plain["metadata"]["weights_hash"]


def test_sequential_export_stamps_weights_hash():
    models = [
        _linear.Linear(receptive_field=2, sample_rate=48_000),
        _linear.Linear(receptive_field=3, sample_rate=48_000),
    ]
    model = _sequential.Sequential(models=models)
    with TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        with open(Path(tmpdir, "model.nam"), "r") as fp:
            model_dict = json.load(fp)
    assert model_dict["metadata"]["weights_hash"] == _ch.weights_hash(model_dict)
    assert model_dict["metadata"]["weights_hash"] == _reference(
        *(child["weights"] for child in model_dict["config"]["models"])
    )


def test_metadata_edits_do_not_change_hash():
    model = _linear.Linear(receptive_field=2, sample_rate=48_000)
    with TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        with open(Path(tmpdir, "model.nam"), "r") as fp:
            model_dict = json.load(fp)
    before = model_dict["metadata"]["weights_hash"]
    model_dict["metadata"].update({"name": "Renamed", "loudness": -12.0})
    assert _ch.weights_hash(model_dict) == before


def test_negative_zero_hashes_as_zero():
    a = {"architecture": "WaveNet", "config": {}, "weights": [-0.0, 1.0]}
    b = {"architecture": "WaveNet", "config": {}, "weights": [0.0, 1.0]}
    assert _ch.weights_hash(a) == _ch.weights_hash(b)


def test_weights_cannot_shift_between_leaves():
    leaf = lambda w: {"architecture": "Linear", "config": {}, "weights": w}
    seq = lambda *ms: {"architecture": "Sequential", "config": {"models": list(ms)}}
    assert _ch.weights_hash(seq(leaf([1.0, 2.0]), leaf([3.0]))) != _ch.weights_hash(
        seq(leaf([1.0]), leaf([2.0, 3.0]))
    )


def test_stamp_skips_an_unhashable_model():
    d = {
        "architecture": "WaveNet",
        "config": {},
        "weights": [float("nan")],
        "metadata": {"weights_hash": "nam-content-v1:stale"},
    }
    assert "weights_hash" not in _ch.stamp_weights_hash(d)["metadata"]
