import random
from typing import Union

import numpy as np
import torch
import torchvision

from segmenter.masks import BaseMask


class FluidMask(BaseMask):
    def __init__(self,num_shapes:int=1, max_radius=40, saturation:float=0.5):
        BaseMask.__init__(self, num_shapes=num_shapes)
        self.num_shapes = num_shapes
        self.max_radius = max_radius
        self.saturation = saturation
        self._check_args()

    def _check_args(self):
        pass

    def _mask2D(self, x:Union[np.ndarray, torch.Tensor])->torch.Tensor:
        """
        Fluid of fog
        Soft edged blobs with partial transparency
        :return: mask as Torch Tensor of size (H, W)
        """
        b, c, h, w = x.shape
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        mask = np.ones((h, w))

        x = random.randint(0, w)
        y = random.randint(0,h)
        radius = random.randint(20, self.max_radius)
        Y, X = np.meshgrid(np.arange(h),
                              np.arange(w), indexing='ij')
        dist = ((X - x)**2 + (Y - y)**2).astype(float)
        blob = np.exp(-dist / (2 * radius**2))
        mask *= 1 - blob  # simulate partial occlusion

        return (np.clip(mask,0, 1) < self.saturation).astype(np.uint8)

if __name__ == '__main__':
    do_show = True
    n_channels = 1
    b, c, h, w = 8, 3, 240, 320
    x_image = torch.randint(0, 1, (b, c, h, w))
    random_mask = FluidMask()
    mask = random_mask(x=x_image)
    assert mask.shape == (b, c, h, w), "Something went wrong, check dimensions."

    trans = torchvision.transforms.ToPILImage()
    out = trans(mask[0])
    out.show()