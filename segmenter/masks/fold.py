from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torchvision

from segmenter.masks import BaseMask


class FoldMask(BaseMask):
    def __init__(self, num_shapes:int=1,
                 max_length:int=80, max_width:int=30, curve_strength:float=0.5):
        BaseMask.__init__(self, num_shapes=num_shapes)
        self.max_length = max_length
        self.max_width = max_width
        self.curve_strength = curve_strength
        self._check_args()


    def _check_args(self):
        pass

    def _mask2D(self, x:Union[np.ndarray, torch.Tensor]):
        """
        Simulates tissue folds as curved occlusions.
        - H, W: image dimensions
        - num_folds: number of folds to simulate
        - max_length: maximum fold length
        - max_width: maximum fold width
        - curve_strength: controls curvature (0 = straight, 1 = strong curve)
        """
        b, c, h, w = x.shape
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        mask = np.zeros((h, w), dtype=np.uint8)

        # Random start and end points
        x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
        dx = np.random.randint(-self.max_length, self.max_length)
        dy = np.random.randint(-self.max_length, self.max_length)
        x1, y1 = np.clip(x0 + dx, 0, w - 1), np.clip(y0 + dy, 0, h - 1)

        # Control point for curvature
        cx = int((x0 + x1) / 2 + self.curve_strength * dy)
        cy = int((y0 + y1) / 2 - self.curve_strength * dx)

        # Generate Bézier curve
        curve = []
        for t in np.linspace(0, 1, 20):
            xt = int((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1)
            yt = int((1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1)
            curve.append((xt, yt))

        # Convert to polygon with thickness
        curve = np.array(curve, dtype=np.int32)
        thickness = np.random.randint(10, self.max_width)
        cv2.polylines(mask, [curve], isClosed=False, color=1, thickness=thickness)

        return mask.astype(float)


if __name__ == '__main__':
    do_show = True
    n_channels = 1
    b, c, h, w = 8, 3, 240, 320
    x_image = torch.randint(0, 1, (b, c, h, w))
    random_mask = FoldMask()
    mask = random_mask(x=x_image)
    assert mask.shape == (b, c, h, w), "Something went wrong, check dimensions."

    trans = torchvision.transforms.ToPILImage()
    out = trans(mask[0])
    out.show()