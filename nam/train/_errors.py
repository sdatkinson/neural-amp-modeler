# File: _errors.py
# Created Date: Saturday April 13th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

"""
"What could go wrong?"
"""

__all__ = ["IncompatibleCheckpointError"]


class IncompatibleCheckpointError(RuntimeError):
    """
    Raised when model loading fails because the checkpoint didn't match the model
    or its hyperparameters
    """

    pass
