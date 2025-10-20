from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torchvision

from segmenter.masks import BaseMask


class RandomShapeMask(BaseMask):
    def __init__(self,
                 num_shapes=1,
                 max_vertices=8,
                 min_radius=5,
                 max_radius=50):
        BaseMask.__init__(self, num_shapes=num_shapes,)
        self.num_shapes = num_shapes
        self.max_vertices = max_vertices
        self.min_radius = min_radius
        self.max_radius = max_radius
        self._check_args()

    def _check_args(self):
        pass

    def _mask2D(self, x:Union[np.ndarray, torch.Tensor]):
        b, c, h, w = x.shape
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        self.centre_x = (self.max_radius, w - self.max_radius)
        self.centre_y = (self.max_radius, h - self.max_radius)

        mask2d = np.zeros((h, w), dtype=np.uint8)
        center_x = np.random.randint(self.centre_x[0], max(self.centre_x[0],
                                                           self.centre_x[1]) + 1)
        center_y = np.random.randint(self.centre_y[0], max(self.centre_y[0],
                                                           self.centre_y[1]) + 1)
        num_vertices = np.random.randint(3, self.max_vertices)
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        radii = np.random.randint(self.min_radius, self.max_radius, size=num_vertices)
        points = np.stack([
            center_x + radii * np.cos(angles),
            center_y + radii * np.sin(angles)
        ], axis=-1).astype(np.int32)
        cv2.fillPoly(mask2d, [points], color=1)
        return mask2d


if __name__ == '__main__':
    do_show = True
    n_channels = 1
    b, c, h, w = 8, 3, 240, 320

    x_image = torch.randint(0, 1, (b, c, h, w))
    random_mask = RandomShapeMask()
    mask = random_mask(x=x_image)
    assert mask.shape == (b, c, h, w), f"Something went wrong: {mask.shape}"

    trans = torchvision.transforms.ToPILImage()
    out = trans(mask[0])
    out.show()
