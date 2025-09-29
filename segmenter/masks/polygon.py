from typing import Tuple

import cv2
import numpy as np
import torchvision

from segmenter.masks import BaseMask


class RandomShapeMask(BaseMask):
    def __init__(self, shape: Tuple,
                 channels: int,
                 num_shapes=2,
                 max_vertices=8,
                 min_radius=10,
                 max_radius=40):
        BaseMask.__init__(self, shape, channels)
        self.num_shapes = num_shapes
        self.max_vertices = max_vertices
        self.min_radius = min_radius
        self.max_radius = max_radius

        self.centre_x = (self.max_radius, self.W - self.max_radius)
        self.centre_y = (self.max_radius, self.H - self.max_radius)
        self._check_args()

    def _check_args(self):
        pass

    def _mask2D(self):
        mask2d = np.zeros((self.H, self.W), dtype=np.uint8)
        center_x = np.random.randint(self.centre_x[0], self.centre_x[1] + 1)
        center_y = np.random.randint(self.centre_y[0], self.centre_y[1] + 1)
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
    random_mask = RandomShapeMask(shape=(b, c, h, w), channels=n_channels)
    mask = random_mask()
    assert mask.shape == (b, n_channels, h, w), f"Something went wrong: {mask.shape}"

    trans = torchvision.transforms.ToPILImage()
    out = trans(mask[0])
    out.show()
