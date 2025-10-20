import abc
from typing import Tuple, Union

import numpy as np
import torch


class BaseMask:
    def __init__(self, num_shapes: int = 1):
        """
        Base class for random mask generation. Will always returns a random mask of shape (B, channels, H, W)
        where B = 1 if len(shape) < 4 & C = 1 if len(shape) == 2.

        So that the returned mask will be immediately ready for batched processing.

        :param num_shapes: The number of shapes to generate in each mask

        """
        self.num_shapes = num_shapes

    @abc.abstractmethod
    def _check_args(self):
        """ Check and unpack arguments including specific arguments in self.kwargs """
        pass

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        mask = np.zeros_like(x, dtype=float)
        for b in range(mask.shape[0]):
            for _ in range(self.num_shapes):
                mask[b] += self._mask2D(x)

        return torch.tensor(mask > 0, dtype=torch.float32)



    forward = __call__

    def mask2d(self, x:Union[np.ndarray, torch.Tensor])-> np.ndarray:
        return self._mask2D(x)

    @abc.abstractmethod
    def _mask2D(self, x:Union[np.ndarray, torch.Tensor])-> np.ndarray:
        """ Implement this method to return a type-specific 2D mask of shape (H, W) """
        pass

    def _expand_channels(self, mask, channels=3):
        return mask.unsqueeze(0).repeat(channels, 1, 1, 1).permute(1, 0, 2, 3)
