# File: content_hash.py
# Created Date: Monday October 26th 2026

"""
`metadata.weights_hash`: a SHA-256 digest that identifies the weights of a `.nam`
file, independent of its metadata.

Two files with the same weights hash run the same model. Editing metadata (name,
author, gear, loudness, ...) does not change the hash, so it can be used to
de-duplicate and sync copies of a model across tools and services.

Definition
----------

::

    weights_hash = "nam-content-v1:" + hex(SHA-256(
        "nam-content-v1" 0x00
        for each leaf model, depth-first:
            u64le(N) || f32le(w[0]) || ... || f32le(w[N-1])
    ))

1. A leaf model is one that owns its weights. Models that hold other models have
   empty top-level `weights`; the hash walks into their children instead:

   * `SlimmableContainer`: `config.submodels`, from the smallest `max_value` to
     the largest.
   * `Sequential`: `config.models`, in processing order.

2. Each weight is read as a number, rounded to a 32-bit IEEE-754 float (the
   precision engines load them at), with -0 written as +0, as 4 little-endian
   bytes. `N` is the leaf's weight count as 8 little-endian bytes, so weights
   can't shift between leaves without changing the hash.
3. The `nam-content-v1` tag names this recipe. A future change to the recipe gets
   a new tag, so hashes from different recipes never look alike.

A model is unhashable, and gets no `weights_hash`, if a leaf has no weights, a
container is malformed (no children, or two submodels with the same
`max_value`), or a weight isn't a finite number that fits in a float32.
"""

import hashlib as _hashlib
import logging as _logging
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Iterator as _Iterator
from typing import List as _List

import numpy as _np

WEIGHTS_HASH_KEY = "weights_hash"
TAG = "nam-content-v1"

_logger = _logging.getLogger(__name__)


class UnhashableModelError(ValueError):
    """
    The model can't be hashed: a leaf with no weights, a malformed container, or a
    weight that isn't a finite float32.
    """


def weights_hash(model_dict: _Dict[str, _Any]) -> str:
    """
    Weights hash of an exported model dictionary (the parsed JSON of a `.nam` file).

    :return: "nam-content-v1:" followed by 64 lowercase hex characters.
    :raises UnhashableModelError: See the module docstring.
    """
    h = _hashlib.sha256(f"{TAG}\0".encode())
    for weights in _leaf_weights(model_dict):
        f = _to_float32(weights)
        h.update(len(f).to_bytes(8, "little"))
        h.update(f.tobytes())
    return f"{TAG}:{h.hexdigest()}"


def stamp_weights_hash(model_dict: _Dict[str, _Any]) -> _Dict[str, _Any]:
    """
    Write `metadata.weights_hash` into the model dictionary (in place) and return
    it. Should be called after any step that can change the weights. An unhashable
    model is written without one.
    """
    metadata = model_dict.setdefault("metadata", {})
    try:
        metadata[WEIGHTS_HASH_KEY] = weights_hash(model_dict)
    except UnhashableModelError as e:
        metadata.pop(WEIGHTS_HASH_KEY, None)
        _logger.warning(f"Not writing {WEIGHTS_HASH_KEY}: {e}")
    return model_dict


def _leaf_weights(model_dict: _Any) -> _Iterator[_List[_Any]]:
    """
    Weight arrays of every leaf model, depth-first, children in hashing order.
    """
    if not isinstance(model_dict, dict):
        raise UnhashableModelError("A model must be an object")
    architecture = model_dict.get("architecture")
    config = model_dict.get("config") or {}
    if not isinstance(config, dict):
        raise UnhashableModelError("config must be an object")
    if architecture == "SlimmableContainer":
        submodels = config.get("submodels")
        if not isinstance(submodels, list) or len(submodels) == 0:
            raise UnhashableModelError("SlimmableContainer has no submodels")
        if any(not isinstance(s, dict) for s in submodels):
            raise UnhashableModelError("SlimmableContainer submodels must be objects")
        if any(not _is_number(s.get("max_value")) for s in submodels):
            raise UnhashableModelError("SlimmableContainer max_value must be a number")
        submodels = sorted(submodels, key=lambda s: s["max_value"])
        max_values = [s["max_value"] for s in submodels]
        if len(set(max_values)) != len(max_values):
            raise UnhashableModelError("SlimmableContainer max_values must be distinct")
        for submodel in submodels:
            yield from _leaf_weights(submodel.get("model"))
    elif architecture == "Sequential":
        models = config.get("models")
        if not isinstance(models, list) or len(models) == 0:
            raise UnhashableModelError("Sequential has no models")
        for child in models:
            yield from _leaf_weights(child)
    else:
        weights = model_dict.get("weights")
        if not isinstance(weights, list) or len(weights) == 0:
            raise UnhashableModelError(f"No weights for architecture {architecture!r}")
        yield weights


def _to_float32(weights: _List[_Any]) -> _np.ndarray:
    # Booleans and strings would coerce silently; other implementations reject them.
    if any(not _is_number(w) for w in weights):
        raise UnhashableModelError("Weights must be numbers")
    try:
        a = _np.asarray(weights, dtype=_np.float64)
    except OverflowError:
        raise UnhashableModelError("Weights must be finite")
    if not _np.isfinite(a).all():
        raise UnhashableModelError("Weights must be finite")
    with _np.errstate(over="ignore"):
        f = a.astype("<f4")
    if not _np.isfinite(f).all():
        raise UnhashableModelError("Weight overflows float32")
    f[f == 0] = 0.0  # -0.0 -> +0.0
    return f


def _is_number(x: _Any) -> bool:
    return type(x) in (int, float)
