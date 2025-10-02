import abc
from typing import Tuple

import numpy as np
import torch


class BaseMask:
    def __init__(self, shape: Tuple, channels: int, num_shapes:int=2):
        """
        Base class for random mask generation. Will always returns a random mask of shape (B, channels, H, W)
        where B = 1 if len(shape) < 4 & C = 1 if len(shape) == 2.

        So that the returned mask will be immediately ready for batched processing.

        :param shape: Shape of the mask (2, 3 or 4 dimensional .. (H,W), (H, W, C) or (B, C, H, W)
        :param channels: The number of output channels (B, H, W, channels)
                         defaults to C if not passed explicitly
        :param num_shapes: The number of shapes to generate in each mask

        """
        if len(shape) == 2:
            self.H, self.W = shape
            self.C = 1
            self.B = 1
        elif len(shape) == 3:
            self.H, self.W, self.C = shape
            self.B = 1
        elif len(shape) == 4:
            self.B, self.C, self.H, self.W = shape
        else:
            raise NotImplementedError("shape must be 2, 3 or 4 dimensional")
        self.channels = channels if channels is not None else self.C
        self.num_shapes = num_shapes


    @abc.abstractmethod
    def _check_args(self):
        """ Check and unpack arguments including specific arguments in self.kwargs """
        pass

    def __call__(self) -> torch.Tensor:
        mask = np.zeros((self.B, self.H, self.W), dtype=float)
        for b in range(self.B):
            for _ in range(self.num_shapes):
                mask[b] += self._mask2D()

        mask = torch.tensor(mask > 0, dtype=torch.float32)
        return self._expand_channels(mask)

    def mask2d(self):
        return self._mask2D()

    @abc.abstractmethod
    def _mask2D(self):
        """ Implement this method to return a type-specific 2D mask of shape (H, W) """
        pass

    def _expand_channels(self, mask):
        return mask.unsqueeze(0).repeat(self.channels, 1, 1, 1).permute(1, 0, 2, 3)
