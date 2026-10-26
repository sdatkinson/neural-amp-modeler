"""
Tests for `metadata.content_hash`
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


def _sha256_of_f32(*weights: float) -> str:
    return hashlib.sha256(struct.pack(f"<{len(weights)}f", *weights)).hexdigest()


def test_plain_model_hashes_little_endian_float32_weights():
    d = {"architecture": "WaveNet", "config": {}, "weights": [0.5, -1.25, 3.0]}
    assert _ch.content_hash(d) == _sha256_of_f32(0.5, -1.25, 3.0)


def test_hash_ignores_metadata_and_config():
    a = {"architecture": "WaveNet", "config": {"a": 1}, "weights": [1.0, 2.0]}
    b = {
        "architecture": "LSTM",
        "config": {"b": 2},
        "metadata": {"name": "x", "loudness": -20.0},
        "weights": [1.0, 2.0],
    }
    assert _ch.content_hash(a) == _ch.content_hash(b)


def test_hash_changes_with_weights():
    a = {"architecture": "WaveNet", "config": {}, "weights": [1.0, 2.0]}
    b = {"architecture": "WaveNet", "config": {}, "weights": [1.0, 2.5]}
    c = {"architecture": "WaveNet", "config": {}, "weights": [2.0, 1.0]}
    assert len({_ch.content_hash(x) for x in (a, b, c)}) == 3


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
    assert _ch.content_hash(container) == _sha256_of_f32(1.0, 2.0, 3.0)


def test_sequential_hashes_children_in_processing_order():
    first = {"architecture": "Linear", "config": {}, "weights": [1.0, 2.0]}
    second = {"architecture": "Linear", "config": {}, "weights": [3.0]}
    seq = {
        "architecture": "Sequential",
        "config": {"models": [first, second]},
        "weights": [],
    }
    assert _ch.content_hash(seq) == _sha256_of_f32(1.0, 2.0, 3.0)


def test_export_stamps_content_hash():
    model = _linear.Linear(receptive_field=2, sample_rate=48_000)
    with TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        with open(Path(tmpdir, "model.nam"), "r") as fp:
            model_dict = json.load(fp)
    stamped = model_dict["metadata"]["content_hash"]
    assert isinstance(stamped, str) and len(stamped) == 64
    # The stamp matches a recomputation from the file as written.
    assert stamped == _ch.content_hash(model_dict)


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

    assert scaled["metadata"]["content_hash"] == _ch.content_hash(from_disk)
    assert scaled["metadata"]["content_hash"] != plain["metadata"]["content_hash"]


def test_sequential_export_stamps_content_hash():
    models = [
        _linear.Linear(receptive_field=2, sample_rate=48_000),
        _linear.Linear(receptive_field=3, sample_rate=48_000),
    ]
    model = _sequential.Sequential(models=models)
    with TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        with open(Path(tmpdir, "model.nam"), "r") as fp:
            model_dict = json.load(fp)
    assert model_dict["metadata"]["content_hash"] == _ch.content_hash(model_dict)
    assert model_dict["metadata"]["content_hash"] == _sha256_of_f32(
        *(w for child in model_dict["config"]["models"] for w in child["weights"])
    )


def test_metadata_edits_do_not_change_hash():
    model = _linear.Linear(receptive_field=2, sample_rate=48_000)
    with TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        with open(Path(tmpdir, "model.nam"), "r") as fp:
            model_dict = json.load(fp)
    before = model_dict["metadata"]["content_hash"]
    model_dict["metadata"].update({"name": "Renamed", "loudness": -12.0})
    assert _ch.content_hash(model_dict) == before
