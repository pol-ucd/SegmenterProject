import random
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, binary_dilation
from torchvision.transforms import v2 as T


COMMON_AUGMENTATIONS = T.Compose([
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # Brightness, Contrast, Saturation, Hue
    T.RandomGrayscale(p=0.2),  # Convert to Grayscale
])
GEOMETRIC_AUGMENTATIONS = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
])


def apply_custom_augmentations(image: torch.Tensor) -> torch.Tensor:
    """Applies common SimSiam-style augmentations (color/geometric) to a single view."""
    return GEOMETRIC_AUGMENTATIONS(COMMON_AUGMENTATIONS(image))


class MaskGenerator:
    """
    Generates synthetic masks to simulate surgical occlusions, instruments,
    and tissue folds for self-supervised pre-training.
    """

    def __init__(self, size: Tuple[int, int] = (512, 512),
                 instrument_ratio: float = 0.3):
        """
        Initializes the mask generator.
        :param size: The target size (H, W) for the square mask.
        :param instrument_ratio: Probability of generating an instrument-like mask.
        """
        self.size = size if len(size) == 2 else (512, 512)
        self.instrument_ratio = instrument_ratio

    def _generate_lesion_mask(self) -> np.ndarray:
        """
        Generates a soft, irregular lesion/fold mask by combining multiple blurred
        ellipses, better mimicking a real polyp or lesion annotation.
        """
        mask = np.zeros(self.size, dtype=np.float32)
        num_shapes = random.randint(1, 3)  # 1 to 3 overlapping shapes

        Y, X = np.ogrid[:self.size[0], :self.size[1]]

        for _ in range(num_shapes):
            sub_mask = np.zeros(self.size, dtype=np.float32)

            # Random parameters for a sub-ellipse
            center_x = random.randint(self.size[0] // 5, self.size[0] * 4 // 5)
            center_y = random.randint(self.size[1] // 5, self.size[1] * 4 // 5)
            radius_x = random.randint(self.size[0] // 15, self.size[0] // 5)
            radius_y = random.randint(self.size[1] // 15, self.size[1] // 5)

            # Create the sub-ellipse
            dist_sq = ((Y - center_y) ** 2 / radius_y ** 2 + (X - center_x) ** 2 / radius_x ** 2)
            sub_mask[dist_sq <= 1] = 1.0

            # Combine with the main mask (Logical OR)
            mask = np.clip(mask + sub_mask, 0.0, 1.0)

        # Apply blurring for organic/smooth boundary
        mask = gaussian_filter(mask, sigma=random.uniform(5, 10))
        # Binarize after blurring to get smooth, irregular boundary
        mask = (mask > random.uniform(0.3, 0.7)).astype(np.float32)

        return mask

    def _generate_instrument_mask(self) -> np.ndarray:
        """Generates a thin, long mask (simulating a surgical instrument)."""
        mask = np.zeros(self.size, dtype=np.float32)

        # Start and end points for a line
        start_x = random.randint(0, self.size[0] - 1)
        start_y = random.randint(0, self.size[1] - 1)
        end_x = random.randint(0, self.size[0] - 1)
        end_y = random.randint(0, self.size[1] - 1)

        # Create a line approximation
        num_points = int(np.hypot(end_x - start_x, end_y - start_y))
        x = np.linspace(start_x, end_x, num_points).astype(int)
        y = np.linspace(start_y, end_y, num_points).astype(int)

        # Clip to boundaries and set instrument path
        x = np.clip(x, 0, self.size[0] - 1)
        y = np.clip(y, 0, self.size[1] - 1)
        mask[y, x] = 1.0

        # Dilate the line to give it thickness
        dilation_structure = np.ones((random.randint(3, 7), random.randint(3, 7)))
        mask = binary_dilation(mask, structure=dilation_structure).astype(np.float32)

        return mask

    def generate_composite_mask(self) -> torch.Tensor:
        """Generates a composite mask based on random and surgical features."""

        # 1. Base Mask (Irregular Lesion/Tissue fold)
        mask = self._generate_lesion_mask()

        # 2. Add Instrument Occlusion (Surgical feature)
        if random.random() < self.instrument_ratio:
            instrument_mask = self._generate_instrument_mask()
            # Combine masks (logical OR)
            mask = np.clip(mask + instrument_mask, 0.0, 1.0)

        return torch.from_numpy(mask).unsqueeze(0).float()  # [1, H, W]

    def create_siamese_pair(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates an Anchor and a Positive image pair using a generated mask AND
        applies custom augmentations.
        :param image: The raw input medical image [C, H, W].
        :return: (Anchor Image, Positive (Occluded) Image)
        """
        # Ensure image is C, H, W
        if image.ndim == 4:  # Assume B=1 if 4D
            image = image.squeeze(0)

        # 1. Apply baseline augmentations to the raw image BEFORE masking
        image_augmented = apply_custom_augmentations(image.clone())

        mask = self.generate_composite_mask()  # [1, H, W]

        # Anchor (View 1) is the fully augmented, but unmasked image.
        anchor = image_augmented

        # Positive (View 2) is the image_augmented *with* occlusion applied.
        # Occlusion mask: 1 - mask
        occlusion_mask = 1.0 - mask
        positive = anchor * occlusion_mask.unsqueeze(0)

        return anchor, positive
