import random

import numpy as np
from segmenter.masks import BaseMask


class FluidMask(BaseMask):
    def __init__(self, max_radius: int = 60, saturation: float = 0.6):
        self.max_radius = max_radius
        self.saturation = saturation
        self._check_args()

    def _check_args(self):
        if not (0.0 < self.saturation <= 1.0):
            raise ValueError("Saturation must be between 0 and 1.")

    def _mask2D(self, h: int, w: int) -> np.ndarray:
        """
        Generates a 2D Gaussian blob with a sharp threshold (for occlusion).
        """
        # 1. Randomized parameters
        x_center = random.randint(0, w)
        y_center = random.randint(0, h)
        radius = random.randint(20, self.max_radius)

        # Vectorized distance calculation
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        dist_sq = ((X - x_center) ** 2 + (Y - y_center) ** 2).astype(float)

        # Gaussian profile
        # Note: We use 1 - blob for "occlusion" or "mask presence"
        blob = np.exp(-dist_sq / (2 * radius ** 2))

        # Thresholding based on saturation
        # The mask is True where the blob is strong (less than 1-saturation)
        mask = blob > (1 - self.saturation)

        return mask.astype(np.uint8)
