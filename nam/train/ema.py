# File: ema.py
# Optional exponential moving average (EMA) of weights for Lightning training.

"""
Optional EMA for training via :class:`pytorch_lightning.callbacks.EMAWeightAveraging`.

Enable in the learning config::

    "ema": {
        "enabled": true,
        "decay": 0.999
    }

When enabled, validation metrics and saved checkpoints use the EMA weights (see Lightning
``WeightAveraging`` / ``EMAWeightAveraging`` docs).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Constructor kwargs for EMAWeightAveraging (``enabled`` is consumed here).
_EMA_KEYS = frozenset(
    {
        "device",
        "use_buffers",
        "decay",
        "update_every_n_steps",
        "update_starting_at_step",
        "update_starting_at_epoch",
    }
)


def ema_callback_from_learning_config(
    learning_config: Dict[str, Any],
) -> Optional[Any]:
    """
    :param learning_config: Same dict as ``nam-full`` / ``full.main`` (may include ``ema``).
    :return: ``EMAWeightAveraging`` instance if ``ema.enabled`` is true, else ``None``.
    """
    ema = learning_config.get("ema")
    if not isinstance(ema, dict) or not ema.get("enabled", False):
        return None
    try:
        from pytorch_lightning.callbacks import EMAWeightAveraging
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "EMA requires pytorch_lightning with EMAWeightAveraging (recent PL 2.x)."
        ) from e

    kwargs = {k: v for k, v in ema.items() if k in _EMA_KEYS}
    return EMAWeightAveraging(**kwargs)
