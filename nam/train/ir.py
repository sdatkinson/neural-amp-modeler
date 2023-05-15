# File: ir.py
# Created Date: Monday February 13th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Code related to fitting impulse responses
"""

from collections import namedtuple
from typing import NamedTuple, Optional

import numpy as np
from scipy.linalg import solve_toeplitz


class Solution(NamedTuple):
    """
    :param solution: (n,)
    :param rxx: (n,) autocorrelation
    :param rxy: (n,) cross-correlation
    """

    solution: np.ndarray
    rxx: np.ndarray
    rxy: np.ndarray


def fit(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    stability: float = 0.1,
    rxx: Optional[np.ndarray] = None,
    rxy: Optional[np.ndarray] = None,
) -> Solution:
    """
    Given input signal x, response y, compute an impulse response of length n

    :param x: Input signal
    :param y: Output signal
    :param n: Length of IR solved for
    :param stability: Increases diagonal of the Toeplitz matrix by this fraction.
    :param rxx: Autocorrelation (if already computed)
    :param rxy: Cross-correlation (if already computed)
    """
    # Uses causal Wiener filter
    # See:
    # https://en.wikipedia.org/wiki/Wiener_filter#Finite_impulse_response_Wiener_filter_for_discrete_series

    assert x.ndim == 1
    assert y.ndim == 1

    def correlation(x1, x2, n):
        assert len(x1) == len(x2)
        num_samples = len(x1)
        return np.array([np.mean(x1[: num_samples - i] * x2[i:]) for i in range(n)])

    rxx = correlation(x, x, n) if rxx is None else rxx
    rxy = correlation(x, y, n) if rxy is None else rxy

    rxx_copy = np.copy(rxx)
    rxx_copy[0] *= 1.0 + stability

    return Solution(solve_toeplitz(rxx_copy, rxy), rxx, rxy)
