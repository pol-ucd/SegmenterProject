import json
from glob import glob
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset, random_split


class PolypDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
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

        return img, mask.unsqueeze(-1).float()  # [1,H,W]


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

def train_val_dataset(dataset, test_split=0.3):
    """
    Split a dataset into training and validation subsets according to the test split value
    :param dataset: A torch.utils.Dataset instance
    :param test_split: percentage of the dataset to use for validation
    :return: (training subset, test subset)
    """
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def data_load(test_split=0.3, batch_size=4) -> tuple[DataLoader[Any], DataLoader[Any]]:

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

    #TODO: Convert file paths into configuration options and/or command line parameters
    train_imgs = sorted(glob('../data/Polyp Segmentation/train/*.jpg'))
    train_jsons = sorted(glob('../data/Polyp Segmentation/train/*.json'))

    test_imgs = sorted(glob('../data/Polyp Segmentation/valid/*.jpg'))
    test_jsons = sorted(glob('../data/Polyp Segmentation/valid/*.json'))

    all_imgs = train_imgs + test_imgs
    all_jsons = train_jsons + test_jsons

    print(f"Found {len(all_imgs)} images and {len(all_jsons)} annotations")

    all_ds = PolypDataset(all_imgs, all_jsons, transforms=None)

    test_len = int(len(all_ds) * test_split)
    lengths = [len(all_ds)  - test_len, test_len]

    train_idx, val_idx = random_split(all_ds, lengths)

    train_ds = PolypSubset(
        train_idx, transform=train_transform
    )

    val_ds = PolypSubset(
        val_idx, transform=valid_transform
    )

    print(f"Found {len(train_ds)} training samples and {len(val_ds)} test samples")

    train_loader: DataLoader[Any] = DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    val_loader: DataLoader[Any] = DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    return train_loader, val_loader
