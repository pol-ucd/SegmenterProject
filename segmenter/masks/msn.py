import random
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, binary_dilation
from torchvision.transforms import v2 as T

from segmenter.masks import FoldMask, InstrumentMask, RandomShapeMask

class Augmentations:
    COMMON = T.Compose([
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # Brightness, Contrast, Saturation, Hue
        T.RandomGrayscale(p=0.2),  # Convert to Grayscale
    ])
    GEOMETRIC = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
    ])


def apply_custom_augmentations(image: torch.Tensor) -> torch.Tensor:
    """Applies common SimSiam-style augmentations (color/geometric) to a single view."""
    return Augmentations.GEOMETRIC(Augmentations.COMMON(image))


class MaskGenerator:
    """
    Generates synthetic masks to simulate surgical occlusions, instruments,
    and tissue folds for self-supervised pre-training.
    """

    def __init__(self,
                 size: Tuple[int, int] = (512, 512),
                 instrument_ratio: float = 0.5,
                 fold_ratio = 0.5,
                 lesion_ratio = 0.5):
        """
        Initializes the mask generator.
        :param size: The target size (H, W) for the square mask.
        :param instrument_ratio: Probability of generating an instrument-like mask.
        """
        self.size = size if len(size) == 2 else (512, 512)
        self.instrument_ratio = instrument_ratio
        self.fold_ratio = fold_ratio
        self.lesion_ratio = lesion_ratio

        """
        num_shapes:int=2,
        max_length:int=80, 
        max_width:int=30, 
        curve_strength:float=0.5
        """
        self.fold = FoldMask(shape=self.size, channels=1)

        """
        num_shapes:int=2, 
        thickness:int=10
        """
        self.instrument = InstrumentMask(shape=self.size, channels=1)

        """
         num_shapes=2,
         max_vertices=8,
         min_radius=10,
         max_radius=40
        """
        self.polygon = RandomShapeMask(shape=self.size, channels=1,)

    @staticmethod
    def _generate_lesion_mask(size: Tuple[int, int]) -> np.ndarray:
        """
        Generates a soft, irregular lesion/fold mask by combining multiple blurred
        ellipses, better mimicking a real polyp or lesion annotation.
        """
        h,w = size
        mask = np.zeros(size, dtype=np.float32)
        num_shapes = random.randint(1, 3)  # 1 to 3 overlapping shapes

        Y, X = np.ogrid[:h, :w]

        for _ in range(num_shapes):
            sub_mask = np.zeros(size, dtype=np.float32)

            # Random parameters for a sub-ellipse
            center_x = random.randint(h // 5, h * 4 // 5)
            center_y = random.randint(w // 5, w * 4 // 5)
            radius_x = random.randint(h // 15, h // 5)
            radius_y = random.randint(w // 15, w // 5)

            # Create the sub-ellipse
            dist_sq = ((Y - center_y) ** 2 / radius_y ** 2 + (X - center_x) ** 2 / radius_x ** 2)
            sub_mask[dist_sq <= 1] = 1.0


            # Combine with the main mask (Logical OR)
            mask = np.clip(mask + sub_mask, 0.0, 1.0)

        mask = MaskGenerator._smooth_mask(mask)

        return mask

    @staticmethod
    def _smooth_mask(mask: np.array) -> np.array:
        # Apply blurring for organic/smooth boundary
        mask = gaussian_filter(mask, sigma=random.uniform(5, 10))
        # Binarize after blurring to get smooth, irregular boundary
        mask = (mask > random.uniform(0.3, 0.7)).astype(np.float32)
        return mask

    def generate_composite_mask(self) -> torch.Tensor:
        """Generates a composite mask based on random and surgical features."""

        # Base Mask (Irregular Lesion/Tissue fold)
        mask = self._generate_lesion_mask(self.size)

        if random.random() < self.lesion_ratio:
            poly_sub_mask = self.polygon.mask2d()
            poly_sub_mask = MaskGenerator._smooth_mask(poly_sub_mask)
            mask = np.clip(mask + poly_sub_mask, 0.0, 1.0)


        if random.random() < self.fold_ratio:
            fold_mask = self.fold.mask2d()
            mask = np.clip(mask + fold_mask, 0, 1)

        # Add Instrument Occlusion (Surgical feature)
        if random.random() < self.instrument_ratio:
            instrument_mask = self.instrument.mask2d()
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

        # Apply baseline augmentations to the raw image BEFORE masking
        image_augmented = apply_custom_augmentations(image.clone())

        mask = self.generate_composite_mask()  # [1, H, W]

        # Anchor (View 1) is the fully augmented, but unmasked image.
        anchor = image_augmented

        # Positive (View 2) is the image_augmented *with* occlusion applied.
        # Occlusion mask: 1 - mask
        occlusion_mask = 1.0 - mask
        positive = anchor * occlusion_mask.unsqueeze(0)

        return anchor, positive

    def create_siamese_triple(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates an Anchor and a Positive image pair AND returns the mask.
        :param image: The raw input medical image [C, H, W].
        :return: (Anchor Image, Positive (Occluded) Image, Mask)
        """
        if image.ndim == 4:
            image = image.squeeze(0)

        image_augmented = apply_custom_augmentations(image.clone())
        mask = self.generate_composite_mask()  # [1, H, W]

        # Anchor (View 1 / Focal)
        anchor = image_augmented

        # Positive (View 2 / Global) - Occlusion applied
        occlusion_mask = 1.0 - mask
        positive = anchor * occlusion_mask.unsqueeze(0)

        # Mask is 1.0 where occlusion/lesion occurs.
        return anchor, positive, mask


if __name__ == '__main__':
    import torchvision
    n_channels = 1
    b, c, h, w = 8, 3, 240, 320
    size = (320, 240)
    instrument_ratio = 0.3
    gen = MaskGenerator(size=size, instrument_ratio=instrument_ratio)

    mask = gen.generate_composite_mask()

    print(mask.shape)

    # assert mask.shape == (b, n_channels, h, w), "Something went wrong, check dimensions."

    trans = torchvision.transforms.ToPILImage()
    out = trans(mask)
    out.show()