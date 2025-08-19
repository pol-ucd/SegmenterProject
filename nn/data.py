import json
import os
import random
from glob import glob
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from config import IMAGE_PATHS, MASK_PATHS, FILE_TYPES


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths,
                 transform=None,
                 use_cutmix=False,
                 cutmix_prob=0.0,
                 num_classes=None,
                 ignore_index=255):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.image_paths)

    def _load_pair(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        print(mask.dtype)
        if mask.max() > 1:
            mask = (mask > 127).astype(int) # expect a (0,1) mask .. some masks have more than two values
        print("mask unique values: ", np.unique(mask))
        return img, mask

    def __getitem__(self, idx):
        img1, m1 = self._load_pair(idx)

        if self.transform:
            img1, m1 = self.transform(img1, m1)

        # Albumentations ToTensorV2 will produce:
        # img1: FloatTensor [C,H,W]; m1: LongTensor [H,W]

        # Optional CutMix for segmentation (hard labels): cut & paste both image and mask
        if self.use_cutmix and random.random() < self.cutmix_prob:
            j = random.randint(0, len(self.image_paths) - 1)
            img2, m2 = self._load_pair(j)

            if self.transform:
                img2, m2 = self.transform(img2, m2)

            C, H, W = img1.shape
            rx, ry, rw, rh = self._rand_bbox(W, H, lam=random.random())
            img1 = img1.clone()
            m1 = m1.clone()

            img1[:, ry:ry + rh, rx:rx + rw] = img2[:, ry:ry + rh, rx:rx + rw]
            m1[ry:ry + rh, rx:rx + rw] = m2[ry:ry + rh, rx:rx + rw]

        return img1, m1.type(torch.LongTensor)  # mask is LongTensor [H,W]

    @staticmethod
    def _rand_bbox(W, H, lam):
        cut_rat = np.sqrt(1.0 - lam)
        rw = max(1, int(W * cut_rat))
        rh = max(1, int(H * cut_rat))
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        rx = np.clip(cx - rw // 2, 0, W - rw)
        ry = np.clip(cy - rh // 2, 0, H - rh)
        return rx, ry, rw, rh


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
