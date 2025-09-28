import random

import numpy as np
import torch
import torchvision

from base_mask import BaseMask


class FluidMask(BaseMask):
    def __init__(self, shape, channels, num_shapes:int=2, max_radius=40):
        BaseMask.__init__(self, shape, channels)
        self.num_shapes = num_shapes
        self.max_radius = max_radius
        self._check_args()

    def _check_args(self):
        pass

    def _mask2D(self)->torch.Tensor:
        """
        Fluid of fog
        Soft edged blobs with partial transparency
        :return: mask as Torch Tensor of size (H, W)
        """
        mask = np.ones((self.H, self.W))
        for _ in range(self.num_shapes):
            x = random.randint(0, self.W)
            y = random.randint(0, self.H)
            radius = random.randint(20, self.max_radius)
            Y, X = np.meshgrid(np.arange(self.H),
                                  np.arange(self.W), indexing='ij')
            dist = ((X - x)**2 + (Y - y)**2).astype(float)
            blob = np.exp(-dist / (2 * radius**2))
            mask *= 1 - blob  # simulate partial occlusion

        return (np.clip(mask,0, 1) < 0.9).astype(np.uint8)

if __name__ == '__main__':
    do_show = True
    n_channels = 1
    b, c, h, w = 8, 3, 240, 320
    random_mask = FluidMask(shape=(b, c, h, w), channels=n_channels)
    mask = random_mask()
    assert mask.shape == (b, n_channels, h, w), "Something went wrong, check dimensions."

    trans = torchvision.transforms.ToPILImage()
    out = trans(mask[0])
    out.show()