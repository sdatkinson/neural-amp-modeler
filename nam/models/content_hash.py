# File: content_hash.py
# Created Date: Monday October 26th 2026

"""
`metadata.content_hash`: a SHA-256 digest that identifies the weights of a `.nam`
file, independent of its metadata.

Two files with the same content hash run the same model. Editing metadata (name,
author, gear, loudness, ...) does not change the hash, so it can be used to
de-duplicate and sync copies of a model across tools and services.

Definition
----------

1. Flatten the model's weights to a sequence of 32-bit IEEE-754 floats (the
   precision at which engines load them) and take their little-endian bytes, in
   the order they appear in the file's `weights` array.
2. Models that hold other models contribute their children's bytes first, in a
   canonical order, followed by their own (`weights`, which is empty for these):

   * `SlimmableContainer`: `config.submodels`, from the smallest `max_value` to
     the largest.
   * `Sequential`: `config.models`, in processing order.

3. `content_hash` is the lowercase hex SHA-256 of the concatenated bytes.
"""

import hashlib as _hashlib
from typing import Any as _Any
from typing import Dict as _Dict

import numpy as _np

CONTENT_HASH_KEY = "content_hash"


def _children(model_dict: _Dict[str, _Any]) -> list:
    """
    Child model dicts, in the order their weights are hashed.
    """
    architecture = model_dict.get("architecture")
    config = model_dict.get("config", {})
    if architecture == "SlimmableContainer":
        submodels = sorted(config["submodels"], key=lambda s: s["max_value"])
        return [s["model"] for s in submodels]
    if architecture == "Sequential":
        return list(config["models"])
    return []


def weights_bytes(model_dict: _Dict[str, _Any]) -> bytes:
    """
    The canonical byte string whose SHA-256 is the content hash.
    """
    own = _np.asarray(model_dict.get("weights", []), dtype="<f4").tobytes()
    return b"".join(weights_bytes(child) for child in _children(model_dict)) + own


def content_hash(model_dict: _Dict[str, _Any]) -> str:
    """
    Content hash of an exported model dictionary (the parsed JSON of a `.nam` file).

    :return: 64 lowercase hex characters.
    """
    return _hashlib.sha256(weights_bytes(model_dict)).hexdigest()


def stamp_content_hash(model_dict: _Dict[str, _Any]) -> _Dict[str, _Any]:
    """
    Write `metadata.content_hash` into the model dictionary (in place) and return
    it. Should be called after any step that can change the weights.
    """
    model_dict.setdefault("metadata", {})[CONTENT_HASH_KEY] = content_hash(model_dict)
    return model_dict
