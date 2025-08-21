"""
Using Advanced Augmentations
----------------------------
Augmentation
- CutMix	Combines two images by cutting and pasting a patch from one into another.
            Labels are mixed proportionally. Improves localization and generalization
- MixUp	    Linearly interpolates between two images and their labels.
            Smooths decision boundaries
- GridMask	Masks out grid-like patches of the image.
            Forces model to learn from partial information
These augmentations are label-aware, so they must be applied inside the Dataset class,
not just as image transforms.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


class TrainingImageTransforms:
    def __init__(self, size: tuple[int, int] = (512, 512)):
        self.tfm = A.Compose([
            # A.RandomResizedCrop(size=size,
            #                     scale=(0.8, 1.0),
            #                     p=0.5),
            A.Resize(height=size[0], width=size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(p=0.2),
            A.Affine(
                scale=(0.9, 1.1),  # ±10% zoom
                translate_percent=((0.05, 0.05)),  # ±5% shift
                rotate=(-15, 15),  # ±15° rotation
                shear=(0.0, 0.0),  # no shear
                fit_output=False,  # keep original size
                border_mode=0,  # cv2.BORDER_CONSTANT
                fill=(0, 0, 0),  # fill color for image
                fill_mask=0,  # fill value for mask
                p=0.5),
            # A.GaussianBlur(p=0.2),
            A.Normalize(),
            ToTensorV2(transpose_mask=True),
        ])

    def __call__(self, image, mask):
        out = self.tfm(image=image, mask=mask)
        return out["image"], out["mask"]  # mask -> [H, W] LongTensor


class ValidationImageTransforms:
    def __init__(self, size: tuple[int, int] = (512, 512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.tfm = A.Compose([
            A.Resize(height=size[0], width=size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(transpose_mask=True),
        ])

    def __call__(self, image, mask):
        out = self.tfm(image=image, mask=mask)
        return out["image"], out["mask"]

    class RescaleMaskTransforms:
        def __init__(self, size: tuple[int, int] = (512, 512)):
            self.tfm = A.Compose([
                A.Resize(height=size[0], width=size[1]),
                # A.Normalize(mean=(0.485, 0.456, 0.406),
                #             std=(0.229, 0.224, 0.225)),
                A.Normalize(),
                ToTensorV2(transpose_mask=True),
            ])

        def __call__(self, image, mask):
            out = self.tfm(image=image, mask=mask)
            return out["image"], out["mask"]

