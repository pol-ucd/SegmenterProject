import pandas as pd

from nn.modules import EarlyStopping

"""
Test script to check the data pipeline works
"""

import numpy as np
import torch
from torch import optim, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.multiclass_jaccard import MulticlassDiceLoss

from nn.data import SegmentationDataset, split_images_and_masks
from nn.models import SegformerBinarySegmentation4
from transforms.images import ValidationImageTransforms, TrainingImageTransforms




class RunConfig:
    def __init__(self, params: dict) -> None:
        self.set_params(params)

    def get_params(self) -> dict:
        return self.params

    def set_params(self, params: dict) -> None:
        self.params = params


class RunManager:
    def __init__(self, **params):
        self.split = params.get("split", "train")
        self.model = params.get("model", None)
        self.criterion = params.get("criterion", None)
        self.optimizer = params.get("optimizer", None)
        self.scheduler = params.get("scheduler", None)
        self.scaler = params.get("scaler", None)
        self.device = params.get("device", "cpu")
        self.train_loader = params.get("train_loader", None)
        self.val_loader = params.get("val_loader", None)
        self.num_classes = params.get("num_classes", 2)
        self.ignore_index = params.get("ignore_index", False)

    def train(self) -> tuple[float, float]:
        self.is_train = True
        return self.step()

    def validate(self) -> tuple[float, float]:
        self.is_train = False
        return self.step()

    def step(self) -> tuple[float, float]:

        loader = self.train_loader if self.is_train else self.val_loader

        self.model.train() if self.is_train else self.model.eval()

        total_loss = 0.0
        total_inter, total_union = (torch.zeros(self.num_classes).to(self.device),
                                    torch.zeros(self.num_classes).to(self.device))

        with torch.set_grad_enabled(self.is_train):
            for imgs, masks in tqdm(loader):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                out = self.model(imgs)  # [B,C,H,W]

                loss = self.criterion(out, masks)

                if self.is_train:
                    self.optimizer.zero_grad()
                    # loss.backward()
                    # self.optimizer.step()

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()  # Fails on MPS, works on CPU/CUDA
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item() * imgs.size(0)

                # IoU metric at eval time (argmax)
                preds = out.argmax(dim=1)  # [B,H,W]

                for c in range(self.num_classes):
                    if c == self.ignore_index: continue
                    pred_c = (preds == c)
                    true_c = (masks == c)
                    inter = (pred_c & true_c).sum().float()
                    union = (pred_c | true_c).sum().float()
                    total_inter[c] += inter
                    total_union[c] += union

        n = len(loader.dataset)
        mean_iou = (total_inter / (total_union + 1e-6))
        mean_iou = mean_iou[total_union > 0].mean().item() if (total_union > 0).any() else 0.0
        return total_loss / n, mean_iou


def main():
    test_split = 0.1
    num_classes = 2  # Binary classification {'not_lesion': 0, 'lesion': 1}
    ignore_index = 255
    batch_size = 4
    num_workers = 0
    n_epochs = 100
    pretained_model = 'nvidia/segformer-b4-finetuned-ade-512-512'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Load the list of training and testing images and masks.
    We take the same percentage split from each separate source of images so
    that we are guaranteed to have a representation from each source in the
    training and testing sets.
    """
    train_images, train_masks, val_images, val_masks = split_images_and_masks(split=test_split)

    df_file = pd.DataFrame({"val_image": val_images,
                            "val_masks": val_masks, })
    df_file.to_csv("validate_files.csv",
                   index=False)

    print(f"Found {len(train_images)} training images and {len(train_masks)} training masks")
    print(f"Found {len(val_images)} testing images and {len(val_masks)} testing masks")

    train_ds = SegmentationDataset(
        train_images, train_masks,
        transform=TrainingImageTransforms(size=(512, 512)),
        use_cutmix=True,
        cutmix_prob=0.2,
        num_classes=num_classes, ignore_index=ignore_index
    )
    val_ds = SegmentationDataset(
        val_images, val_masks,
        transform=ValidationImageTransforms(size=(512, 512)),
        # use_cutmix=False,
        num_classes=num_classes, ignore_index=ignore_index
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    model = SegformerBinarySegmentation4(pretrained_model=pretained_model,
                                         num_classes=num_classes).to(device)
    model.to(device)

    # criterion = CEJaccardLoss(num_classes=num_classes, ignore_index=ignore_index, ce_weight=0.2)
    criterion = MulticlassDiceLoss(num_classes=num_classes, ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10, T_mult=2)
    """
    Only use GradScaler if we have CUDA
    """
    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()

    step_params = {"model": model, "optimizer": optimizer, "criterion": criterion, "device": device,
                   "scheduler": scheduler, "scaler": scaler,
                   "stage": "train", "train_loader": train_loader, "val_loader": val_loader, "epoch": n_epochs,
                   "num_classes": num_classes, "ignore_index": ignore_index}
    runner = RunManager(**step_params)

    best_score = 0
    early_stopper = EarlyStopping(patience=7, min_delta=0.001, mode='max', verbose=True,
                                  save_path="best_model_classica.pt")
    for epoch in range(n_epochs):

        tr_loss, tr_miou = runner.train()
        va_loss, va_miou = runner.validate()

        print(
            f"Epoch {epoch:02d} | Train Loss {tr_loss:.4f} mIoU {tr_miou:.3f} | Val Loss {va_loss:.4f} mIoU {va_miou:.3f}")

        early_stopper(va_miou, model, epoch)

        if early_stopper.early_stop:
            print(f"Training stopped early at epoch {epoch}")
            break


if __name__ == "__main__":
    main()
