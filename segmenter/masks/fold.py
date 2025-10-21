import cv2
import numpy as np

from segmenter.masks import BaseMask


class FoldMask(BaseMask):
    def __init__(self, max_length: int = 100, max_width: int = 30, curve_strength: float = 0.5):
        self.max_length = max_length
        self.max_width = max_width
        self.curve_strength = curve_strength
        self._check_args()

    def _check_args(self):
        pass

    def _mask2D(self, h: int, w: int) -> np.ndarray:
        """
        Simulates tissue folds as curved occlusions using OpenCV for drawing Bézier curves.
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        # Random start and end points
        x0, y0 = np.random.randint(0, w), np.random.randint(0, h)

        # Ensure dx/dy don't exceed max_length while keeping the point in bounds
        dx = np.random.randint(-self.max_length, self.max_length)
        dy = np.random.randint(-self.max_length, self.max_length)

        x1 = np.clip(x0 + dx, 0, w - 1)
        y1 = np.clip(y0 + dy, 0, h - 1)

        # Control point for curvature (requires numpy integer types)
        # Note: Added * w/h factors to curve strength to make it scale better with image size
        scaled_curve_strength = self.curve_strength * min(h, w) / 100.0

        cx = int((x0 + x1) / 2 + scaled_curve_strength * dy)
        cy = int((y0 + y1) / 2 - scaled_curve_strength * dx)

        # Generate Bézier curve points (fewer points for faster drawing)
        curve = []
        for t in np.linspace(0, 1, 15):
            # Quadratic Bézier: P(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
            xt = int((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1)
            yt = int((1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1)
            curve.append((xt, yt))

        # Convert to polygon points and draw with thickness using OpenCV
        curve_pts = np.array(curve, dtype=np.int32)
        thickness = np.random.randint(5, self.max_width)

        # Use cv2.polylines to draw a thick, curved line
        cv2.polylines(mask, [curve_pts], isClosed=False, color=1, thickness=thickness)

        return mask
