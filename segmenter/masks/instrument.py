import random

import numpy as np
import torchvision

from segmenter.masks import BaseMask


class InstrumentMask(BaseMask):
    """
    Instrument occlusion mask.
    Long, thin rectangles or curves (e.g. forceps, scopes)
    :param H: Height (int)
    :param W: Width (
    :param num_strips: How many instruments to include
    :param thickness: Thickness of the instrument
    :return: mask as Torch Tensor of size (H, W)
    """
    def __init__(self, shape, channels, num_shapes:int=2, thickness:int=10):
        BaseMask.__init__(self, shape, channels)
        self.num_shapes = num_shapes
        self.thickness = thickness
        self._check_args()

    def _check_args(self):
        if self.num_shapes <= 0:
            raise ValueError("num_shapes must be positive")
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")

    def _mask2D(self)->np.array:
        """
            Creates a 2D mask (H, W) with a line of given thickness between two points.
        """
        x1, y1 = random.randint(0, self.W - 1), random.randint(0, self.H - 1)
        x2, y2 = random.randint(0, self.W - 1), random.randint(0, self.H - 1)
        start_point = (x1, y1)
        end_point = (x2, y2)

        # Generate all (x, y) coordinates of the mask
        # We use np.mgrid to get coordinate arrays (Y, X)
        Y, X = np.mgrid[0:self.H, 0:self.W]

        # Convert to floating point for calculations
        X = X.astype(float)
        Y = Y.astype(float)

        # Convert start and end points to NumPy arrays (x, y)
        P1 = np.array(start_point)  # (x1, y1)
        P2 = np.array(end_point)  # (x2, y2)

        # Vector from P1 to P2
        V = P2 - P1

        # Vector from P1 to all pixels (X, Y)
        P_minus_P1_x = X - P1[0]
        P_minus_P1_y = Y - P1[1]

        # Create array of dot products V . (P - P1)
        # V . (P - P1) = V_x * (X - P1_x) + V_y * (Y - P1_y)
        dot_product = V[0] * P_minus_P1_x + V[1] * P_minus_P1_y

        # Squared length of V
        V_sq_len = np.sum(V ** 2)

        if V_sq_len == 0:
            # Handle the case where start_point == end_point (a single point)
            dist_sq = (X - P1[0]) ** 2 + (Y - P1[1]) ** 2
            mask = dist_sq < (self.thickness / 2.0) ** 2
            return mask

        # Calculate parameter t: the projection of P onto the line segment P1P2
        # t = (V . (P - P1)) / |V|^2
        t = dot_product / V_sq_len

        # Clamp t to [0, 1] to ensure we only consider the segment, not the infinite line
        t_clamped = np.clip(t, 0.0, 1.0)

        # Find the closest point (P_closest) on the segment to each pixel (X, Y)
        # P_closest = P1 + t_clamped * V
        P_closest_x = P1[0] + t_clamped * V[0]
        P_closest_y = P1[1] + t_clamped * V[1]

        # Calculate the squared distance from each pixel (X, Y) to P_closest
        dist_sq = (X - P_closest_x) ** 2 + (Y - P_closest_y) ** 2

        # Create the mask
        # Pixels are part of the line if their distance is less than or equal to half the thickness
        half_thickness = self.thickness / 2.0
        mask = dist_sq <= half_thickness ** 2

        return mask



if __name__ == '__main__':
    n_channels = 1
    b, c, h, w = 8, 3, 240, 320
    random_mask = InstrumentMask(shape=(b, c, h, w), channels=n_channels)
    mask = random_mask()
    assert mask.shape == (b, n_channels, h, w), "Something went wrong, check dimensions."

    trans = torchvision.transforms.ToPILImage()
    out = trans(mask[0])
    out.show()

