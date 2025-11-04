import argparse
import glob
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Union, List

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations import ToTensorV2
from torch import autocast, nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from segmenter.loss import IoULoss
from segmenter.loss.hybrid import HybridLoss

torch.multiprocessing.set_sharing_strategy('file_system')

WEIGHTS_MAP = {'base': None,
               'mim': '../segmenter/checkpoint/attention_mim_segformer_pretrained.pt'}

# test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
test_sizes = [0.9]
# backbone = "nvidia/segformer-b4-finetuned-ade-512-512"
backbone = "nvidia/segformer-b5-finetuned-ade-640-640"
hdf5_file = '../segmenter/data/Classica.h5'
prefix = 'attention_verify_classica_images'
image_size = 256
checkpoint_path = "../segmenter/checkpoint/attention_mim_segformer_pretrained.pt"
n_folds = 10

n_augments = 3
n_batch = 4
path_to_training_images = '../segmenter/data/training'
path_to_validation_images = '../segmenter/data/validation'


def check_scores(metric: dict[str, list]) -> bool:
    all_lens = np.array([len(v) for v in metric.values()])
    base_len = all_lens[0]
    if not np.all(all_lens == base_len):
        for k, v in metric.items():
            print(f"{k}: {len(v)}")
    return np.all(all_lens == base_len)



def get_image_mask_pairs(root_dir: str = None, image_types: List[str] = None):
    if root_dir is None:
        raise ValueError("root_dir cannot be None. Please specify a valid root directory.")
    image_types = image_types or ['*.jpg', '*.png', '*.tiff']
    image_dirs = sorted([x[0] for x in os.walk(root_dir) if x[0].endswith('images')])
    mask_dirs = sorted([x[0] for x in os.walk(root_dir) if x[0].endswith('masks')])
    try:
        assert len(image_dirs) == len(mask_dirs)
    except:
        msg = (f'Image and Mask directories do not match. '
               f'Images: {image_dirs} and masks: {mask_dirs}')
        raise ValueError(msg)

    result = []
    for image_dir, mask_dir in zip(image_dirs, mask_dirs):
        for image_type in image_types:
            images = sorted(glob.glob(os.path.join(image_dir, image_type)))
            masks = sorted(glob.glob(os.path.join(mask_dir, image_type)))
            result += [(i, m) for i, m in zip(images, masks)]
    return result

def collate_fn(batch):
    """
    batch: list of items from Dataset
           Each item is a list of (image_tensor, mask_tensor) tuples.
    """
    imgs, masks = [], []
    for pairs in batch:
        for img, mask in pairs:
            imgs.append(img)
            masks.append(mask)
    return torch.stack(imgs), torch.stack(masks)

class ImageMaskDataset(Dataset):
    def __init__(self, root_dir, n_augment=2, image_size=(256, 256)):
        """
        Args:
            root_dir (str): Path to root directory containing images/masks subdirectories.
            n_augment (int): Number of random augmentations per image-mask pair.
            image_size (tuple): Resize dimensions for images and masks.
        """


        self.n_augment = n_augment

        self.mask_image_pairs = get_image_mask_pairs(root_dir=root_dir)

        self.image_files,  self.mask_files = zip(*self.mask_image_pairs)
        assert len(self.image_files) == len(self.mask_files), "Mismatch in image and mask count."

        # Base transform (resize + tensor conversion)
        self.base_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Augmentation pipeline (rich Albumentations transforms)
        self.augment_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.02, 0.05), rotate=(-30, 30),
                     shear=(-10, 10), p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):


        # Load image and mask
        img = np.array(Image.open(self.image_files[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_files[idx]).convert("L"))  # grayscale mask

        # Original pair
        base = self.base_transform(image=img, mask=mask)
        pairs = [(base['image'], base['mask'].unsqueeze(0))]  # add channel to mask

        # Augmented pairs
        for _ in range(self.n_augment):
            augmented = self.augment_pipeline(image=img, mask=mask)
            augmented = self.base_transform(image=augmented['image'], mask=augmented['mask'])
            pairs.append((augmented['image'], augmented['mask'].unsqueeze(0)))

        return pairs  # list of (image, mask) tensors




class SimpleMaskSegmenter(torch.nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path: Union[str, Any],
                 config: SegformerConfig,
                 load_dict_path: Union[str, Any] = None,
                 num_classes: int = 1):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.load_dict_path = load_dict_path
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ignore_mismatched_sizes=True)

        if self.load_dict_path is not None:
            self.load_model(self.load_dict_path)

        self.segmenter_head = nn.Linear(config.num_labels, num_classes)

    def forward(self, pixel_map):
        b, c, h, w = pixel_map.shape
        out = self.model(pixel_map)[0]
        out = out.permute(0, 2, 3, 1)
        out = self.segmenter_head(out)
        out = out.permute(0, 3, 1, 2)

        out = F.interpolate(out, size=(h, w), mode="nearest-exact")
        assert out.shape == (b, self.num_classes,h, w), f"{__class__: }: Size mismatch between image and segmenter output"
        return out

    def load_model(self, path: str):
        device = next(self.model.parameters()).device
        state_dict = torch.load(path, weights_only=False,
                                map_location=device)
        self.model.segformer.load_state_dict(state_dict )


class EpochStopper:
    def __init__(self, max_boredom: int = 5, min_delta: float = None):
        self.best_loss = float('inf')
        self.min_delta = min_delta or 0.0001
        self.max_boredom = max_boredom
        self.current_boredom = 0

    def __call__(self, epoch: int, score: np.floating[Any]) -> bool:
        """ Set up stopping criteria - stop after 'boredom' steps do not improve loss by 'min_delta' """
        is_stopping = False
        if score + self.min_delta < self.best_loss:
            self.best_loss = score
            self.boredom = 0
        else:
            self.boredom += 1

        if self.boredom >= self.max_boredom:
            is_stopping = True
        return is_stopping

    forward = __call__


def main(params: dict[str, Any]):
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prefix = 'attention_verify_classica'
    image_size = (256, 256)
    learning_rate = params['learning_rate']
    prefix = params['prefix']


    num_epochs = params['num_epochs']
    batch_size = params['batch_size'] or 4
    num_workers = params['num_workers']

    config = SegformerConfig.from_pretrained(backbone)

    metrics = {"case": [],
               "test_split": [],
               "test_iteration": [],
               "is_test": [],
               "dice": [],
               "iou": [],
               "precision": [],
               "recall": []}

    results_csv_path = os.path.join(Path.home(), "segmenter")
    results_csv_name = os.path.join(results_csv_path, f"{prefix}_results.csv")
    results = pd.DataFrame.from_dict(metrics)
    results.to_csv(results_csv_name)

    split_loss = []

    train_dataset = ImageMaskDataset(root_dir=path_to_training_images, n_augment=n_augments)
    val_dataset = ImageMaskDataset(root_dir=path_to_validation_images, n_augment=0)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                collate_fn=collate_fn)

    model = SimpleMaskSegmenter(pretrained_model_name_or_path=backbone,
                                config=config,
                                load_dict_path=None,
                                num_classes=1)

    model.to(device=device)

    loss_params = {
        "ce": {
            "weight": 0.1
        },
        "dice": {
            "weight": 0.0
        },
        "focal": {
            "weight": 0.0,
            "alpha": 0.25,
            "gamma": 2.0
        },
        "tversky": {
            "weight": 0.0,
            "alpha": 0.8,
            "beta": 0.2
        },
        "iou": {
            "weight": 1.0
        },
        "boundary_sdf": {
            "weight": 0.0,
            "dt_backend": "kornia"
        },
        "soft_chamfer": {
            "weight": 0.0,
            "dt_backend": "kornia"
        }
    }
    loss_fn = HybridLoss(loss_params)

    # Only pass the parameters that require gradients to the optimizer
    optimizer = torch.optim.AdamW(
        # params=filter(lambda p: p.requires_grad, model.parameters()),
        params=model.parameters(),
        lr=learning_rate,
        # weight_decay=l2_decay_penalty  # L2 regularization to prevent large weights
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    """
    Only use GradScaler if we have CUDA
    """
    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()

    stopper = EpochStopper(max_boredom=3, min_delta=0.0001)

    train_names = []
    test_names = []
    for epoch_idx, epoch in enumerate(range(num_epochs)):
        logger.info(f"Epoch [{epoch_idx + 1}/{num_epochs}].")
        total_epoch_train_loss = []
        total_epoch_test_loss = []
        iou_epoch_train_loss = []
        iou_epoch_test_loss = []

        for imgs, masks in train_dataloader:
            model.train()

            imgs = imgs.to(device=device)
            masks = masks.to(device=device)

            optimizer.zero_grad()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                seg_map = model(imgs)

                assert seg_map.shape == masks.shape, (f"Size mismatch between "
                                                           f"generated mask: {seg_map.shape}, "
                                                           f"and target mask: {masks.shape}")

                loss_train = loss_fn(seg_map, masks)

            if scaler is not None:
                scaler.scale(loss_train).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            total_epoch_train_loss += [loss_train.item()]
            with torch.no_grad():
                iou_epoch_train_loss += [IoULoss()(seg_map, masks).item()]

            b, _, _, _ = seg_map.shape

        for imgs, masks in val_dataloader:
            model.eval()

            with torch.no_grad():
                mask_out = model(imgs)
                loss_test = loss_fn(mask_out, masks)

            total_epoch_test_loss += [loss_test.item()]

            iou_epoch_test_loss += [IoULoss()(mask_out, masks).item()]

        scheduler.step()
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. "
                    f"Training Total Loss: {np.mean(total_epoch_train_loss):.4f} "
                    f"Training IoU Loss: {np.mean(iou_epoch_train_loss):.4f} "
                    f"Test Total Loss: {np.mean(total_epoch_test_loss):.4f} "
                    f"Test IoU Loss: {np.mean(iou_epoch_test_loss):.4f} ")

        if stopper(epoch=epoch, score=np.mean(total_epoch_train_loss)):
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed. ")
            logger.info(f"Training set: {train_names}")
            split_loss += [stopper.best_loss]
            break
        else:
            if stopper.boredom > 0:
                logger.info(f"No obvious improvement. Current boredom level is {stopper.boredom} / {stopper.max_boredom}.")


def get_args():
    """
    Command line arguments

    :return: Dictionary of arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=hdf5_file,
                        type=str, help="Path to the HDF5 file.")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, )
    parser.add_argument("-nw", "--num_workers", type=int, default=4, )
    parser.add_argument("-e", "--num_epochs", type=int, default=200, )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, )
    parser.add_argument("-p", "--prefix", type=str, default=prefix, )
    parser.add_argument("-ro", "--run_once", type=bool, default=False, )
    parser.add_argument("-ns", "--num_shapes", type=int, default=24, )

    args = parser.parse_args()

    params = {'dataset': args.input,
              'batch_size': args.batch_size,
              'num_workers': args.num_workers,
              'num_epochs': args.num_epochs,
              'learning_rate': args.learning_rate,
              'prefix': args.prefix,
              'run_once': bool(args.run_once),
              'num_shapes': int(args.num_shapes), }

    return params


if __name__ == "__main__":
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        force=True,  # Resets any previous configuration - in Colab for example
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )
    logger = logging.getLogger()
    try:
        params = get_args()
        main(params)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down gracefully.")
    except Exception as ex:
        logger.error(f"Unknown exception occurred. Error: {ex}")
        logger.error(traceback.format_exc())
    finally:
        # ensure log handlers are flushed.
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.info("Logger handlers flushed and closed. Exiting now.")
        sys.exit(0)
