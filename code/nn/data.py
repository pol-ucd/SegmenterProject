import json
from glob import glob
from typing import Any

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class PolypDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

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

        return img, mask.unsqueeze(0).float()  # [1,H,W]


def data_load(test_split=0.3) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:

    train_transform = A.Compose([A.Resize(512, 512),
                                 A.HorizontalFlip(p=0.5),
                                 A.RandomBrightnessContrast(p=0.4),
                                 A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                                 A.GaussianBlur(p=0.2),
                                 A.Normalize(),
                                 ToTensorV2()])

    valid_transform = A.Compose([A.Resize(512, 512),
                                 A.Normalize(),
                                 ToTensorV2()])

    #TODO: Convert file paths into configuration options and/or command line parameters
    train_imgs = sorted(glob('../data/Polyp Segmentation/train/*.jpg'))
    train_jsons = sorted(glob('../data/Polyp Segmentation/train/*.json'))

    valid_imgs = sorted(glob('../data/Polyp Segmentation/valid/*.jpg'))
    valid_jsons = sorted(glob('../data/Polyp Segmentation/valid/*.json'))

    print(f"Found {len(train_imgs)} training images")
    print(f"Found {len(train_jsons)} training JSON")
    print(f"Found {len(valid_imgs)} validation images")
    print(f"Found {len(valid_jsons)} validation JSON")

    all_imgs = train_imgs + valid_imgs
    all_jsons = train_jsons + valid_jsons

    (train_imgs, temp_imgs,
     train_jsons, temp_jsons) = train_test_split(all_imgs, all_jsons,
                                                 test_size=test_split,
                                                 random_state=42)

    (val_imgs, test_imgs,
     val_jsons, test_jsons) = train_test_split(temp_imgs, temp_jsons,
                                               test_size=stest_split,
                                               random_state=42)

    train_ds = PolypDataset(train_imgs, train_jsons, train_transform)
    val_ds = PolypDataset(val_imgs, val_jsons, valid_transform)
    test_ds = PolypDataset(test_imgs, test_jsons, valid_transform)

    train_loader: DataLoader[Any] = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
