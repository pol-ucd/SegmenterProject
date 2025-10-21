import random

import numpy as np

from segmenter.masks import BaseMask


class InstrumentMask(BaseMask):
    def __init__(self, thickness: int = 20):
        self.thickness = thickness
        self._check_args()

    def _check_args(self):
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")

    def _mask2D(self, h: int, w: int) -> np.ndarray:
        """
        Creates a 2D mask (H, W) with a straight line segment of given thickness.
        Uses pure NumPy vector math for performance.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Random endpoints
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        P1 = np.array((x1, y1), dtype=float)
        P2 = np.array((x2, y2), dtype=float)

        # 2. Coordinate mesh and vectors
        Y, X = np.mgrid[0:h, 0:w].astype(float)
        V = P2 - P1  # Segment vector (Vx, Vy)

        V_sq_len = np.sum(V ** 2)

        if V_sq_len < 1e-6:  # Handle point case or very short segment
            dist_sq = (X - P1[0]) ** 2 + (Y - P1[1]) ** 2
            mask[dist_sq < (self.thickness / 2.0) ** 2] = 1
            return mask

        # 3. Projection onto the line (t)
        P_minus_P1_x = X - P1[0]
        P_minus_P1_y = Y - P1[1]
        dot_product = V[0] * P_minus_P1_x + V[1] * P_minus_P1_y
        t = dot_product / V_sq_len

        # 4. Clamp t to [0, 1] for segment only
        t_clamped = np.clip(t, 0.0, 1.0)

        # 5. Closest point on segment
        P_closest_x = P1[0] + t_clamped * V[0]
        P_closest_y = P1[1] + t_clamped * V[1]

        # 6. Squared distance to segment
        dist_sq = (X - P_closest_x) ** 2 + (Y - P_closest_y) ** 2

        # 7. Apply mask
        half_thickness = self.thickness / 2.0
        mask[dist_sq <= half_thickness ** 2] = 1

        return mask

