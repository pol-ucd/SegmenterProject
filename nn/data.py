import logging
import os
import random
from datetime import datetime
from glob import glob
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


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
    image_paths = image_paths
    mask_paths = mask_paths
    file_types = file_types

    logger = logging.getLogger(__name__)
    if file_types is None or image_paths is None or mask_paths is None:
        logger.error(f"Error loading data: {file_types}, {image_paths} and {mask_paths} cannot be None")
        raise ValueError(f"Either image_paths or mask_paths must be provided.")

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


class CheckpointError(Exception):
    """
    Custom exception for errors related to loading or saving CSV files
    within the CSVHandler class.
    """

    def __init__(self, message="An error occurred with the CSV file operation."):
        self.message = message
        super().__init__(self.message)


class CheckpointHandler:
    """
    A class to handle loading and saving pandas DataFrames from CSV files.

    This class provides methods to load a CSV file, save the DataFrame to a
    CSV file, and includes error handling using a custom exception class.
    """

    def __init__(self, save_file_path: str=None, load_file_path: str=None,
                 suffix: str = None, prefix: str=None):
        """
        Initializes the CheckpointHandler with an option
        al file path.

        Args:
            save_file_path (str, optional): The path to the checkpoint file to be loaded.
                                       If provided, it will attempt to load the data.
            load_file_path (str, optional): The path to the checkpoint file to be loaded.
            suffix (str, optional): The optional datetime suffix appended to the filename
                                      If provided it is appended at save time and used as
                                      a suffix to search for a checkpoint at load time.
                                      Defaults to None in which case a system generated
                                      datetime suffix is used for load and save.
        """

        self.prefix = prefix if prefix is not None else "model_checkpoint"
        self.suffix = suffix
        if suffix is None:
            # Get the current date and time
            now = datetime.now()
            # YYYY-MM-DD_HH-MM-SS
            self.suffix = now.strftime("%Y-%m-%d_%H-%M-%S")

        self.df = pd.DataFrame()  # Initialize an empty DataFrame
        self.pt = None
        self.train_images, self.train_masks, self.test_images, self.test_masks = None, None, None, None

        self.save_file_path = save_file_path
        self.load_file_path = load_file_path
        if load_file_path:
            self.load()

    def load(self):
        """
        Loads a CSV file into the class's DataFrame attribute.

        Args:
            file_path (str, optional): The path to the CSV file. If not provided,
                                       it uses the file_path from initialization.
        """
        file_name = self.prefix + "_" + self.suffix
        pt_name = os.path.join(self.load_file_path, file_name + ".pt")
        csv_name = os.path.join(self.load_file_path, file_name + ".csv")
        if not os.path.isfile(pt_name) or not os.path.isfile(csv_name):
            raise CheckpointError(f"Checkpoint not found or path is invalid: {self.load_file_path}")

        try:
            self.df = pd.read_csv(csv_name)
            self.pt = torch.load(pt_name)
        except pd.errors.EmptyDataError:
            self.df = pd.DataFrame()  # Reset DataFrame
            raise CheckpointError(f"The file {self.load_file_path} is empty.")
        except pd.errors.ParserError as e:
            self.df = pd.DataFrame()
            raise CheckpointError(f"Unable to parse the file {self.load_file_path}. Check its format. Original error: {e}")
        except Exception as e:
            self.df = pd.DataFrame()
            raise CheckpointError(f"An unexpected error occurred during loading: {e}")
        self.train_images = self.df[self.df.phase == "T"]["image"].values
        self.train_masks = self.df[self.df.phase == "T"]["mask"].values
        self.val_images = self.df[self.df.phase == "V"]["image"].values
        self.val_masks = self.df[self.df.phase == "V"]["mask"].values

    def save(self, index: bool=False):
        """
        Saves the current DataFrame to a CSV file.

        Args:
            index (bool): Whether to write the DataFrame index to the CSV.
                          Defaults to False.
        """
        file_name = self.prefix + "_" + self.suffix
        pt_name = os.path.join(self.save_file_path, file_name + ".pt")
        csv_name = os.path.join(self.save_file_path, file_name + ".csv")

        # Ensure the directory exists before saving
        directory = os.path.dirname(self.save_file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                raise CheckpointError(f"Error creating directory {directory}: {e}")

        try:
            self.df.to_csv(csv_name, index=index)
            torch.save(self.pt, pt_name)
        except Exception as e:
            raise CheckpointError(f"An error occurred while saving the Checkpoint: {e}")

