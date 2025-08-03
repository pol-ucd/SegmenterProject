import json
import os
from glob import glob
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


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
