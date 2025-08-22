import logging
import os
import random
from glob import glob
from typing import Any, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

import config


# --- Custom Paired Augmentation Classes ---
# These classes ensure the same random transformation is applied to both the image and the mask.

class PairedRandomHorizontalFlip:
    """
    Applies a random horizontal flip to both the image and the mask.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask


class PairedRandomRotation:
    """
    Applies a random rotation to both the image and the mask.
    """

    def __init__(self, degrees, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            angle = transforms.RandomRotation.get_params(self.degrees)
            # Use interpolation=Image.NEAREST for the mask to prevent new pixel values.
            img = F.rotate(img, angle, interpolation=Image.BICUBIC)
            mask = F.rotate(mask, angle, interpolation=Image.NEAREST)
        return img, mask


class PairedColorJitter:
    """
    Applies a color jitter transformation to the image only.
    The mask remains unchanged as it contains segmentation labels.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = self.transform(img)
        return img, mask


class PairedRandomCropAndResize:
    """
    Applies a random crop to both the image and the mask, then resizes
    the cropped regions back to the original size.
    """

    def __init__(self, size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=self.scale, ratio=self.ratio
            )
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            img = F.resize(img, self.size, interpolation=Image.BICUBIC)
            mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
        return img, mask


# --- Custom Dataset Class for Semantic Segmentation Augmentation ---

class SemanticSegmentationDatasetAugmentor(Dataset):
    """
    A custom PyTorch Dataset subclass for semantic segmentation
    that loads image-mask pairs and applies N random augmentations
    for each original pair, effectively boosting the dataset size.
    """

    def __init__(self, image_paths, mask_paths, n_augments, image_size=(512, 512),
                 mean: Tuple = (0.485, 0.456, 0.406),
                 std: Tuple = (0.229, 0.224, 0.225)):
        """
        Initializes the dataset.

        Args:
            image_paths (list): A list of paths to the images.
            mask_paths (list): A list of paths to the masks.
            n_augments (int): The number of augmented pairs to generate for each original pair.
            image_size (tuple): The size to which images and masks will be resized.
            mean (Tuple): Mean for image normalization.
            std (Tuple): Standard deviation for image normalization.
        """
        assert len(image_paths) == len(mask_paths)
        self.image_files = image_paths
        self.mask_files = mask_paths
        self.N = n_augments
        self.img_mean = mean
        self.img_std = std

        assert len(self.image_files) == len(self.mask_files), "Image and mask lists must have the same number of files."

        # Define base transformations that are always applied
        self.base_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Define the series of random augmentations
        self.augmentations = [
            PairedRandomRotation(degrees=(-45, 45), p=0.5),
            PairedRandomHorizontalFlip(p=0.5),
            PairedColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            PairedRandomCropAndResize(size=image_size, p=0.5)
        ]

        # Define a single resize transformation
        self.resize_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
        ])

        self.transform_norm = transforms.Compose([
            transforms.Normalize(self.img_mean, self.img_std)
        ])

    def __len__(self):
        """
        Returns the total number of samples in the dataset, including augmented ones.
        Each original pair contributes (N + 1) samples (N augmented + 1 original).
        """
        return len(self.image_files) * (self.N + 1)

    def __getitem__(self, idx):
        """
        Retrieves a single image-mask pair from the dataset.
        The index determines if an original or an augmented pair is returned.
        """
        original_idx = idx // (self.N + 1)
        version_idx = idx % (self.N + 1)

        image_path = self.image_files[original_idx]
        mask_path = self.mask_files[original_idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.resize_transform(image)
        mask = self.resize_transform(mask)

        if version_idx > 0:
            for transform in self.augmentations:
                image, mask = transform(image, mask)

        image = self.base_transforms(image)
        mask = self.base_transforms(mask)

        # Ensure mask contains only 0 and 1 values. Masks with >127 are considered 1.
        if mask.max() > 1:
            mask = (mask > 127).long()

        image = self.transform_norm(image)
        return image, mask.long()


class SemanticSegmentationDatasetBasic(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=(512, 512),
                 mean: Tuple = (0.485, 0.456, 0.406),
                 std: Tuple = (0.229, 0.224, 0.225)):
        self.image_files = image_paths
        self.mask_files = mask_paths
        self.size = image_size
        self.img_mean = mean
        self.img_std = std

        assert len(image_paths) == len(mask_paths)

        self.image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Ensure mask contains only 0 and 1 values.
        if mask.max() > 1:
            mask = (mask > 127).long()

        return image, mask.long()


def split_images_and_masks(image_paths: list = None,
                           mask_paths: list = None,
                           file_types: list = None,
                           split: float = None) -> tuple[Any, Any, Any, Any]:
    """
    Splits image and mask file paths into training and testing sets.

    Args:
        image_paths (list): A list of directories containing images.
        mask_paths (list): A list of directories containing masks.
        file_types (list): A list of file extensions to search for (e.g., "*.jpg").
        split (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple[Any, Any, Any, Any]: A tuple containing four lists:
            (train_images, train_masks, test_images, test_masks)
    """
    split_size = split if split is not None else 0.1
    image_paths = config.IMAGE_PATHS if image_paths is None else image_paths
    mask_paths = config.MASK_PATHS if mask_paths is None else mask_paths
    file_types = config.FILE_TYPES if file_types is None else file_types
    logger = logging.getLogger(__name__)

    all_images, all_masks = [], []
    train_idx, test_idx = [], []

    # Iterate through specified paths to find all image and mask files.
    for img_path, mask_path in zip(image_paths, mask_paths):
        for file_type in file_types:
            images_found = sorted(glob(os.path.join(img_path, file_type)))
            masks_found = sorted(glob(os.path.join(mask_path, file_type)))

            start_idx = len(all_images)
            all_images.extend(images_found)
            all_masks.extend(masks_found)

            if len(images_found) > 0:
                indices = np.arange(len(images_found)) + start_idx
                _, _, _train, _test = train_test_split(indices[:, np.newaxis], indices,
                                                       test_size=split_size,
                                                       shuffle=False)
                train_idx.extend(_train)
                test_idx.extend(_test)

    # Sanity check: ensure image and mask basenames match.
    for idx, image in enumerate(all_images):
        base_name = os.path.splitext(os.path.basename(image))[0]
        # Remove any suffix like '_mask' for proper comparison
        base_name = base_name.split("_")[0]
        try:
            assert os.path.splitext(os.path.basename(all_masks[idx]))[0].startswith(base_name)
        except AssertionError:
            logger.error(
                f"Assertion error: image with {base_name} does not match with mask: {os.path.basename(all_masks[idx])}")

    all_images, all_masks = np.array(all_images), np.array(all_masks)
    logger.info(f"Found {len(all_images)} total image-mask pairs.")

    return (all_images[train_idx], all_masks[train_idx],
            all_images[test_idx], all_masks[test_idx])

