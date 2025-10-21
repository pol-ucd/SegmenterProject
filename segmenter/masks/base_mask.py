import abc

import numpy as np



class BaseMask(abc.ABC):
    """
    Base class for random mask generation.
    It enforces the contract that subclasses must implement _mask2D(h, w),
    which returns a single, randomized 2D mask of shape (H, W).
    """

    def __init__(self):
        """No longer uses num_shapes here; composition is handled by CompositeMask."""
        self._check_args()

    @abc.abstractmethod
    def _check_args(self):
        """ Check and unpack arguments including specific arguments in self.kwargs """
        pass

    @abc.abstractmethod
    def _mask2D(self, h: int, w: int) -> np.ndarray:
        """
        Implement this method to return a type-specific 2D mask of shape (H, W).
        Returns: np.ndarray (H, W) of dtype uint8 or bool.
        """
        pass

