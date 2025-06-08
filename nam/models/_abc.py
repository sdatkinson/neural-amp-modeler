"""
Abstractions
"""

import abc as _abc
from typing import Sequence as _Sequence


# Once this is implemented for all nets, put it as part of BaseNet
class ImportsWeights(_abc.ABC):
    @_abc.abstractmethod
    def import_weights(self, weights: _Sequence[float]):
        """
        Assign the provided weights to the model.

        :param weights: 1D array of weights to assign.
        """
        pass
