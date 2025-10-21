import cv2
import numpy as np

from segmenter.masks import BaseMask

class RandomShapeMask(BaseMask):
    def __init__(self, max_vertices: int = 12, min_radius: int = 5, max_radius: int = 120):
        self.max_vertices = max_vertices
        self.min_radius = min_radius
        self.max_radius = max_radius
        self._check_args()

    def _check_args(self):
        pass

    def _mask2D(self, h: int, w: int) -> np.ndarray:
        """
        Generates a randomized polygonal shape mask using OpenCV's fillPoly.
        """
        mask2d = np.zeros((h, w), dtype=np.uint8)

        # Define bounds to ensure the shape doesn't spawn right on the edge
        padding = self.max_radius

        # Center point
        center_x = np.random.randint(padding, max(w - padding, padding + 1))
        center_y = np.random.randint(padding, max(h - padding, padding + 1))

        # Random polygon parameters
        num_vertices = np.random.randint(3, self.max_vertices + 1)

        # Generate uneven angles and radii for irregular shapes
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
        radii = np.random.randint(self.min_radius, self.max_radius + 1, size=num_vertices)

        # Calculate polygon points
        points = np.stack([
            center_x + radii * np.cos(angles),
            center_y + radii * np.sin(angles)
        ], axis=-1).astype(np.int32)

        # Draw filled polygon
        cv2.fillPoly(mask2d, [points], color=1)

        return mask2d

