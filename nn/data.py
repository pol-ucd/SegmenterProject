import json
import os
import random
from glob import glob
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from config import IMAGE_PATHS, MASK_PATHS, FILE_TYPES


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
            # Fix: Replaced 'resample' with 'interpolation' argument.
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
            # Get parameters for the random crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=self.scale, ratio=self.ratio
            )
            # Apply the crop
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

            # Resize the cropped regions back to the original size
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
                 mean:tuple=(0.485, 0.456, 0.406), #Default is to use ImageNet mean & std
                 std:tuple=(0.229, 0.224, 0.225)):
        """
        Initializes the dataset.

        Args:
            image_paths (list): A list of paths to the images.
            mask_paths (list): A list of paths to the masks.
            n_augments (int): The number of augmented pairs to generate for each original pair.
            image_size (tuple): The size to which images and masks will be resized.
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
            transforms.ToTensor(),  # Converts PIL Image to Tensor
        ])

        # Define the series of random augmentations
        self.augmentations = [
            PairedRandomRotation(degrees=(-45, 45), p=0.5),
            PairedRandomHorizontalFlip(p=0.5),
            PairedColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.1,
                              p=0.5),
            PairedRandomCropAndResize(size=image_size,
                                      p=0.5)  # Added the new augmentation
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
        # Determine the index of the original file and the augmentation version
        original_idx = idx // (self.N + 1)
        version_idx = idx % (self.N + 1)

        # Get the file paths for the original image and mask
        image_path = self.image_files[original_idx]
        mask_path = self.mask_files[original_idx]

        # Load the image and mask using Pillow
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # L mode for single-channel mask

        # Resize both image and mask consistently
        image = self.resize_transform(image)
        mask = self.resize_transform(mask)

        # Apply augmentations if not the original version
        if version_idx > 0:
            # Iterate through the list of augmentations and apply them one by one
            for transform in self.augmentations:
                image, mask = transform(image, mask)

        # Apply base transformations to both
        image = self.base_transforms(image)
        mask = self.base_transforms(mask)
        if mask.max() > 1:
            # Expect a (0,1) mask .. some masks have more than two values especially
            # if saved in a lossy format like jpeg or compressed png. TIFF seems to
            # ork best for lossless masks
            mask = (mask > 127).astype(int)

        image = self.transform_norm(image)
        mask = self.transform_norm(mask)
        return image, mask.long()



# --- For testing ----
# To run this example, you need to create dummy image and mask directories.

def create_dummy_data(image_dir, mask_dir, num_pairs=5):
    """
    Helper function to create dummy image and mask files.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    dummy_image = Image.new('RGB', (512, 512), color='red')
    dummy_mask = Image.new('L', (512, 512), color='blue')

    for i in range(num_pairs):
        dummy_image.save(os.path.join(image_dir, f'image_{i}.png'))
        dummy_mask.save(os.path.join(mask_dir, f'mask_{i}.png'))

    print(f"Created {num_pairs} dummy image-mask pairs.")



"""
PyTorch Dataset implementation 
Assumes masks are available as images (i.e. already extracted if in RF Archive)

returns (image, mask) pairs 
"""
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths,
                 transform=None,
                 num_classes=None,
                 ignore_index=255):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def _load_pair(self, idx):
        # img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        # Load the image and mask using Pillow
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # L mode for single-channel mask

        img = np.array(img)
        mask = np.array(mask)

        if mask.max() > 1:
            # Expect a (0,1) mask .. some masks have more than two values especially
            # if saved in a lossy format like jpeg or compressed png. TIFF seems to
            # ork best for lossless masks
            mask = (mask > 127).astype(int)
        return img, mask

    def __getitem__(self, idx):
        img1, m1 = self._load_pair(idx)

        if self.transform:
            img1, m1 = self.transform(img1, m1)

        # img1: FloatTensor [C,H,W] (torch.float32) ; m1: LongTensor [H,W] (torch.int32)
        return img1, m1.type(torch.LongTensor)  # mask is LongTensor [H,W]


class PolypDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None, device=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def set_transform(self, transform):
        if transform is not None:
            self.transforms = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        with open(self.mask_paths[idx], 'r') as f:
            data = json.load(f)

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in data['annotations']:
            mask = np.maximum(mask, maskUtils.decode(ann['segmentation']))

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        else:
            img, mask = torch.tensor(img), torch.tensor(mask)

        return img.float(), mask.unsqueeze(-1).long()  # [1,H,W]


class PolypSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, mask = self.subset[index]
        if self.transform:
            augmented = self.transform(image=img.numpy(), mask=mask.numpy())
            img, mask = augmented['image'], augmented['mask']
        return img, mask

    def __len__(self):
        return len(self.subset)


"""
Load the data requested and return training and validation DataLoader objects split according
to the test_split value

:returns (train_loader, val_loader)
"""


def data_load(data_path,
              test_split,
              batch_size,
              verbose=False) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_transform = A.Compose([A.Resize(512, 512),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.4),
                                 A.ShiftScaleRotate(shift_limit=0.05,
                                                    scale_limit=0.1,
                                                    rotate_limit=15,
                                                    p=0.5),
                                 A.GaussianBlur(p=0.2),
                                 A.Normalize(),
                                 ToTensorV2()])

    valid_transform = A.Compose([A.Resize(512, 512),
                                 A.Normalize(),
                                 ToTensorV2()])

    all_imgs = sorted(glob(os.path.join(data_path, '*.jpg')))
    all_jsons = sorted(glob(os.path.join(data_path, '*.json')))

    train_loader, val_loader = None, None
    n_train, n_val = 0, 0

    if 0.0 < test_split < 1.0:
        train_imgs, test_imgs, train_jsons, test_jsons = train_test_split(all_imgs,
                                                                          all_jsons,
                                                                          test_size=test_split,
                                                                          random_state=42
                                                                          )

        train_ds = PolypDataset(train_imgs, train_jsons, transforms=train_transform)

        val_ds = PolypDataset(test_imgs, test_jsons, transforms=valid_transform)

        train_loader: DataLoader[Any] = DataLoader(train_ds, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)
        val_loader: DataLoader[Any] = DataLoader(val_ds, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        n_train, n_val = len(train_ds), len(val_ds)

    elif test_split <= 0.0:
        train_ds = PolypDataset(all_imgs, all_jsons, transforms=train_transform)
        train_loader: DataLoader[Any] = DataLoader(train_ds, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)
        val_loader: None
        n_train = len(train_ds)
    else:
        val_ds = PolypDataset(all_imgs, all_jsons, transforms=valid_transform)
        val_loader: DataLoader[Any] = DataLoader(val_ds, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        train_loader = None

        n_val = len(val_ds)

    if verbose:
        print(f"Found {n_train} training samples and {n_val} test samples")

    return train_loader, val_loader


def data_load_all(train_path,
                  valid_path,
                  test_split,
                  batch_size,
                  verbose=False) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    train_transform = A.Compose([A.Resize(512, 512),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.4),
                                 A.ShiftScaleRotate(shift_limit=0.05,
                                                    scale_limit=0.1,
                                                    rotate_limit=15,
                                                    p=0.5),
                                 A.GaussianBlur(p=0.2),
                                 A.Normalize(),
                                 ToTensorV2()])

    valid_transform = A.Compose([A.Resize(512, 512),
                                 A.Normalize(),
                                 ToTensorV2()])

    train_imgs = sorted(glob(os.path.join(train_path, '*.jpg')))
    train_jsons = sorted(glob(os.path.join(train_path, '*.json')))
    valid_imgs = sorted(glob(os.path.join(valid_path, '*.jpg')))
    valid_jsons = sorted(glob(os.path.join(valid_path, '*.json')))

    all_imgs = train_imgs + valid_imgs
    all_jsons = train_jsons + valid_jsons
    train_imgs, temp_imgs, train_jsons, temp_jsons = train_test_split(all_imgs, all_jsons, test_size=0.3,
                                                                      random_state=42)
    val_imgs, test_imgs, val_jsons, test_jsons = train_test_split(temp_imgs, temp_jsons, test_size=1 / 3,
                                                                  random_state=42)

    train_ds = PolypDataset(train_imgs, train_jsons, train_transform)
    val_ds = PolypDataset(val_imgs, val_jsons, valid_transform)
    test_ds = PolypDataset(test_imgs, test_jsons, valid_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def data_load_train_test(train_path,
                         valid_path,
                         test_split=0.2,
                         batch_size=4,
                         verbose=False) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_transform = A.Compose([A.Resize(512, 512),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.4),
                                 A.ShiftScaleRotate(shift_limit=0.05,
                                                    scale_limit=0.1,
                                                    rotate_limit=15,
                                                    p=0.5),
                                 A.GaussianBlur(p=0.2),
                                 A.Normalize(),
                                 ToTensorV2()])

    valid_transform = A.Compose([A.Resize(512, 512),
                                 A.Normalize(),
                                 ToTensorV2()])

    train_imgs = sorted(glob(os.path.join(train_path, '*.jpg')))
    train_jsons = sorted(glob(os.path.join(train_path, '*.json')))
    valid_imgs = sorted(glob(os.path.join(valid_path, '*.jpg')))
    valid_jsons = sorted(glob(os.path.join(valid_path, '*.json')))

    all_imgs = train_imgs + valid_imgs
    all_jsons = train_jsons + valid_jsons
    train_imgs, val_imgs, train_jsons, val_jsons = train_test_split(all_imgs, all_jsons,
                                                                    test_size=test_split,
                                                                    random_state=42)

    train_ds = PolypDataset(train_imgs, train_jsons, train_transform)
    val_ds = PolypDataset(val_imgs, val_jsons, valid_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


MAGE_PATHS = [
    # "data/Classica/images",
    "data/Polyp Segmentation/train",
    "data/Polyp Segmentation/valid"]

MASK_PATHS = [
    # "data/Classica/masks",
    "data/Polyp Segmentation/train_masks",
    "data/Polyp Segmentation/valid_masks"]

FILE_TYPES = ["*.jpg", "*.png", "*.jpeg"]


def split_images_and_masks(image_paths: list = None,
                           mask_paths: list = None,
                           file_types: list = None,
                           split: float = None) -> tuple[Any, Any, Any, Any]:
    split_size = split if split is not None else 0.1
    image_paths = IMAGE_PATHS if image_paths is None else image_paths
    mask_paths = MASK_PATHS if mask_paths is None else mask_paths
    file_types = FILE_TYPES if file_types is None else file_types

    all_images, all_masks = [], []
    train_idx, test_idx = [], []
    for image_path, mask_path in zip(image_paths, mask_paths):
        for file_type in file_types:
            images_found = sorted(glob(os.path.join(image_path, file_type)))
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

    """ A quick sanity check to see if the images and masks are in the same order """
    for idx, image in enumerate(all_images):
        base_name = os.path.splitext(os.path.basename(image))[0].split("_")[0]
        try:
            assert os.path.splitext(os.path.basename(all_masks[idx]))[0].startswith(base_name)
        except AssertionError:
            print(f"Assertion error: image with {base_name} does not match with image[idx]")

    all_images, all_masks = np.array(all_images), np.array(all_masks)

    return (all_images[train_idx], all_masks[train_idx],
            all_images[test_idx], all_masks[test_idx])
