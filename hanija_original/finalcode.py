# -*- coding: utf-8 -*-


import os, cv2, json
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss

# 1. Dataset Class
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

        with open(self.mask_paths[idx]) as f:
            data = json.load(f)

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in data['annotations']:
            mask = np.maximum(mask, maskUtils.decode(ann['segmentation']))

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        return img, mask[None, :, :].float()

# 2. Transforms
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

# 3. Paths
train_imgs = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/train/*.jpg'))
train_jsons = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/train/*.json'))
valid_imgs = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/valid/*.jpg'))
valid_jsons = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/valid/*.json'))

all_imgs = train_imgs + valid_imgs
all_jsons = train_jsons + valid_jsons
train_imgs, temp_imgs, train_jsons, temp_jsons = train_test_split(all_imgs, all_jsons, test_size=0.3, random_state=42)
val_imgs, test_imgs, val_jsons, test_jsons = train_test_split(temp_imgs, temp_jsons, test_size=1/3, random_state=42)

# 4. Dataloaders
train_ds = PolypDataset(train_imgs, train_jsons, train_transform)
val_ds = PolypDataset(val_imgs, val_jsons, valid_transform)
test_ds = PolypDataset(test_imgs, test_jsons, valid_transform)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

# 5. Model
from transformers import SegformerConfig

config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
config.num_labels = 1
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512", config=config, ignore_mismatched_sizes=True)
model.classifier = nn.Sequential(
    nn.Conv2d(config.hidden_sizes[-1], 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 1, kernel_size=1)
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 6. Loss and Optimizer
bce = nn.BCEWithLogitsLoss()
tversky = TverskyLoss(mode='binary', alpha=0.2, beta=0.4)
focal = FocalLoss(mode='binary')

def combined_loss(preds, targets):
    if preds.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(targets, size=preds.shape[-2:], mode='nearest')
    return 0.5 * bce(preds, targets) + 0.3 * tversky(torch.sigmoid(preds), targets) + 0.2 * focal(torch.sigmoid(preds), targets)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler = GradScaler()

# 7. Metrics
def compute_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    TP = (preds * targets).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()
    eps = 1e-7
    return {
        'dice': (2 * TP / (2 * TP + FP + FN + eps)).item(),
        'iou': (TP / (TP + FP + FN + eps)).item(),
        'precision': (TP / (TP + FP + eps)).item(),
        'recall': (TP / (TP + FN + eps)).item()
    }

# 8. Train and Evaluate

def train_one_epoch(loader):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader):
        images, masks = images.to(model.device), masks.to(model.device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(pixel_values=images, labels=None)
            loss = combined_loss(outputs.logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    total = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(model.device), masks.to(model.device)
            outputs = model(pixel_values=images)
            preds = torch.sigmoid(outputs.logits)
            if preds.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(masks, size=preds.shape[-2:], mode='nearest')
            metrics = compute_metrics(preds, masks)
            for k in total:
                total[k] += metrics[k]
    avg = {k: total[k] / len(loader) for k in total}
    print(f"Dice: {avg['dice']:.4f} | IoU: {avg['iou']:.4f} | Precision: {avg['precision']:.4f} | Recall: {avg['recall']:.4f}")
    return avg

# 9. Training Loop
best_dice = 0.0
for epoch in range(1, 201):
    print(f"\nEpoch {epoch}/200")
    loss = train_one_epoch(train_loader)
    val_metrics = evaluate(val_loader)
    scheduler.step(epoch + 1)
    print(f"Train Loss: {loss:.4f}")
    if val_metrics['dice'] > best_dice:
        best_dice = val_metrics['dice']
        torch.save(model.state_dict(), "best_segformer.pth")
        print("Saved best model!")

# 10. Inference
model.load_state_dict(torch.load("best_segformer.pth"))
model.eval()

def infer_and_visualize(model, loader, num_samples=5):
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(model.device)
            outputs = model(pixel_values=images)
            preds = (torch.sigmoid(outputs.logits) > 0.5).float().cpu().numpy()
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            masks = masks.cpu().numpy()

            for i in range(images.shape[0]):
                if shown >= num_samples:
                    return
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(images[i])
                ax[0].set_title("Input Image")
                ax[1].imshow(masks[i][0], cmap='gray')
                ax[1].set_title("Ground Truth")
                ax[2].imshow(preds[i][0], cmap='gray')
                ax[2].set_title("Prediction")
                for a in ax:
                    a.axis("off")
                plt.tight_layout()
                plt.show()
                shown += 1

print("\nSample Predictions on Test Set:")
infer_and_visualize(model, test_loader, num_samples=5)

print("\nFinal Evaluation on Test Set:")
test_metrics = evaluate(test_loader)
print(f"Test Dice: {test_metrics['dice']:.4f}")
print(f"Test IoU: {test_metrics['iou']:.4f}")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")

"""# **GRAD-CAM**"""

import os, cv2, json
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils
from transformers import SegformerModel

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

        return img, mask.unsqueeze(0).float()

train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

train_imgs = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/train/*.jpg'))
train_jsons = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/train/*.json'))
valid_imgs = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/valid/*.jpg'))
valid_jsons = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/valid/*.json'))

all_imgs = train_imgs + valid_imgs
all_jsons = train_jsons + valid_jsons

train_imgs, temp_imgs, train_jsons, temp_jsons = train_test_split(all_imgs, all_jsons, test_size=0.3, random_state=42)
val_imgs, test_imgs, val_jsons, test_jsons = train_test_split(temp_imgs, temp_jsons, test_size=1/3, random_state=42)

train_ds = PolypDataset(train_imgs, train_jsons, train_transform)
val_ds = PolypDataset(val_imgs, val_jsons, valid_transform)
test_ds = PolypDataset(test_imgs, test_jsons, valid_transform)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

class SegformerBinarySegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        hidden_dim = self.backbone.config.hidden_sizes[-1]
        self.decode_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, pixel_values):
        features = self.backbone(pixel_values=pixel_values).last_hidden_state
        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=pixel_values.shape[2:], mode='bilinear', align_corners=False)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegformerBinarySegmentation().to(device)

from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss
bce = nn.BCEWithLogitsLoss()
tversky = TverskyLoss(mode='binary', alpha=0.2, beta=0.4)
focal = FocalLoss(mode='binary')

def combined_loss(preds, targets):
    if preds.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(targets, size=preds.shape[-2:], mode='nearest')
    return 0.5 * bce(preds, targets) + 0.3 * tversky(torch.sigmoid(preds), targets) + 0.2 * focal(torch.sigmoid(preds), targets)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler = GradScaler()

def train_one_epoch(loader):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(pixel_values=images)
            loss = combined_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader), None  # No Dice

def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(pixel_values=images)
            loss = combined_loss(logits, masks)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Val Loss: {avg_loss:.4f}")
    return avg_loss

# --- Training loop with loss only ---
num_epochs = 200
best_loss = float('inf')

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, _ = train_one_epoch(train_loader)
    print(f"Train Loss: {train_loss:.4f}")
    val_loss = evaluate(val_loader)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_segformer.pth")
        print("Model saved!")

# --- Grad-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, inputs, class_idx=None):
        self.model.zero_grad()
        outputs = self.model(inputs)
        target = outputs[:, 0, :, :].sum()
        target.backward(retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
        activations = self.activations[0]

        for i in range(pooled_gradients.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        heatmap = cv2.resize(heatmap.numpy(), (inputs.size(3), inputs.size(2)))

        return heatmap

# --- Grad-CAM visualization ---
model.eval()
target_layer = model.decode_head[0]
grad_cam = GradCAM(model, target_layer)

def show_gradcam_on_image(img_tensor, heatmap):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * img + 0.5 * heatmap
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(overlay)
    plt.show()

for images, masks in test_loader:
    images = images.to(device)
    for i in range(images.size(0)):
        input_img = images[i].unsqueeze(0)
        heatmap = grad_cam(input_img)
        show_gradcam_on_image(input_img[0], heatmap)
    break  # Only one batch

"""# **Heatmaps**"""

# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import torch.nn as nn
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import griddata

# ========== 1. Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
config.num_labels = 1
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",
    config=config,
    ignore_mismatched_sizes=True
)
model.classifier = nn.Sequential(
    nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 1, kernel_size=1)
)
model.load_state_dict(torch.load("/home/user/Documents/hanija/fracture_model.pt", map_location=device))
model.to(device)
model.eval()

# ========== 2. Parameters ==========
transform = Compose([Resize(512, 512), Normalize(), ToTensorV2()])
nRows, nCols = 6, 6
duration_after_detection = 30  # seconds
fps = 60

# ========== 3. Manual Input ==========
try:
    user_input_time = float(input("Enter start time (in seconds) for white light detection: "))
except ValueError:
    print("Invalid input. Using default of 5.0 seconds.")
    user_input_time = 5.0

def calculateMeanIntensities(frame, nRows, nCols):
    J3 = np.split(frame, nCols, axis=1)
    J3 = np.array(J3)
    J3 = np.mean(J3, 2, keepdims=True)
    J3 = np.squeeze(J3)
    J3 = np.ravel(J3)
    J3 = np.reshape(J3, (nCols, nRows, -1))
    J3 = np.transpose(J3, (0, 2, 1))
    J3 = np.mean(J3, 1, keepdims=True)
    J3 = np.transpose(np.squeeze(J3), (1, 0))
    return J3

def features(time, data, nRows, nCols, speckleMask):
    m, n = data.shape
    time = np.array(time)

    dIramp = 0.5
    dTramp = 45
    kramp = np.ones((m,1), dtype=np.int64)
    kpeak = np.ones((m,1), dtype=np.int64)
    khalf = np.ones((m,1), dtype=np.int64)
    Ibase = np.full((m,1), np.nan)
    Ipeak = np.full((m,1), np.nan)

    for i, row in enumerate(data):
        diff = (row - row[0]) > dIramp
        if np.any(diff):
            krmp = np.argmax(diff)
            kramp[i] = krmp
            Ibase[i] = row[krmp]
            idx = (time <= (time[krmp] + dTramp))
            if np.any(idx):
                kpeak[i] = np.argmax(row[idx])
                Ipeak[i] = row[kpeak[i]]
                Ihalf = Ibase[i] + (Ipeak[i] - Ibase[i]) / 2
                rng = np.arange(kramp[i], kpeak[i] + 1)
                if len(rng) > 0:
                    khalf[i] = np.argmin(np.abs(row[rng] - Ihalf)) + rng[0] - 1

    tRamp = time[kramp.flatten()]
    dThalf = time[khalf.flatten()] - time[kramp.flatten()]
    dTpeak = time[kpeak.flatten()] - time[kramp.flatten()]
    ingress = np.divide(Ipeak.flatten(), dTpeak.flatten(), out=np.full_like(Ipeak.flatten(), np.nan), where=dTpeak.flatten()!=0)

    def interpolateFeature(fGrid, mask):
        x, y = np.meshgrid(np.arange(fGrid.shape[1]), np.arange(fGrid.shape[0]))
        points = np.column_stack((x[mask], y[mask]))
        values = fGrid[mask]
        grid_x, grid_y = np.mgrid[0:fGrid.shape[0], 0:fGrid.shape[1]]
        return griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.nan)

    timeToPeak = interpolateFeature(np.transpose(np.reshape(dTpeak, (nCols, nRows)), (1, 0)), speckleMask)
    timeToHalf = interpolateFeature(np.transpose(np.reshape(dThalf, (nCols, nRows)), (1, 0)), speckleMask)
    maxIntensity = interpolateFeature(np.transpose(np.reshape(Ipeak.flatten(), (nCols, nRows)), (1, 0)), speckleMask)
    maxIngress = interpolateFeature(np.transpose(np.reshape(ingress, (nCols, nRows)), (1, 0)), speckleMask)

    featuresTable = pd.DataFrame(np.stack((tRamp, dThalf, dTpeak, Ibase.flatten(), Ipeak.flatten()), axis=1),columns=["tRamp", "dThalf", "dTpeak", "Ibase", "Ipeak"])
    return featuresTable, timeToPeak, timeToHalf, maxIntensity, maxIngress

def plot_feature_maps(timeToPeak, timeToHalf, maxIntensity, maxIngress):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    cmap = 'hot'
    sigma = 0.5
    zoom_factor = 2

    def smooth_and_zoom(data):
        smoothed = gaussian_filter(data, sigma=sigma)
        return zoom(smoothed, zoom=zoom_factor, order=3)

    smoothed = [smooth_and_zoom(x) for x in [maxIntensity, maxIngress, timeToHalf, timeToPeak]]
    titles = ["Maximum Intensity", "Maximum Ingress", "Time to 50% Max Intensity", "Time to Peak"]

    for ax, data, title in zip(axs.ravel(), smoothed, titles):
        im = ax.imshow(data, cmap=cmap, interpolation='bicubic', origin='upper')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = "/home/user/Documents/hanija/feature_maps.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Feature map plot saved to {save_path}")

# ========== 5. Process Video ==========
cap = cv2.VideoCapture("/home/user/Documents/hanija/sample.mov")
frame_count = 0
detected_time_sec = None
downscaled_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    if detected_time_sec is None and current_time_sec >= user_input_time:
        detected_time_sec = user_input_time
        print(f"Manual start time reached at {current_time_sec:.2f}s (frame {frame_count})")

    if detected_time_sec is not None:
        elapsed = current_time_sec - detected_time_sec
        if elapsed > duration_after_detection:
            print("30 seconds of data collected. Done.")
            break

        h, w, _ = frame.shape
        left_w = w // 4
        top = h // 3
        bottom = 2 * h // 3
        roi = frame[top:bottom, 0:left_w]

        augmented = transform(image=roi)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(pixel_values=input_tensor)
            pred_mask = torch.sigmoid(output.logits)[0, 0].cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
            pred_mask = cv2.resize(pred_mask, (left_w, bottom - top))

        overlay = roi.copy()
        colored_mask = np.zeros_like(overlay)
        colored_mask[:, :, 1] = pred_mask
        blended = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
        small_overlay = cv2.resize(blended, (96, 72), interpolation=cv2.INTER_AREA)

        downscaled_frames.append(small_overlay)

cap.release()

# ========== 6. Extract Features & Plot ==========
gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in downscaled_frames]
frame_stack = np.stack(gray_frames)

Iregion = []
for f in frame_stack:
    mean_intensities = calculateMeanIntensities(f, nRows, nCols)
    Iregion.append(mean_intensities)

Iregion = np.array(Iregion)
Iregion = np.transpose(Iregion, (1, 2, 0))
Iregion = np.reshape(Iregion, (nRows * nCols, -1))

times = np.linspace(0, duration_after_detection, Iregion.shape[1])
speckleMask = np.ones((nRows, nCols), dtype=bool)

# Optional debug plot
for i in range(min(5, Iregion.shape[0])):
    plt.plot(times, Iregion[i], label=f"Region {i}")
plt.title("Sample Region Intensities Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Intensity")
plt.legend()
plt.grid()
plt.show()

featuresTable, timeToPeak, timeToHalf, maxIntensity, maxIngress = features(times, Iregion, nRows, nCols, speckleMask)
print(featuresTable)

plot_feature_maps(timeToPeak, timeToHalf, maxIntensity, maxIngress)

# ========== 7. Save Overlay Video ==========
output_path = "/home/user/Documents/hanija/overlay_output.mp4"
height, width, _ = downscaled_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in downscaled_frames:
    out.write(frame)

out.release()
print(f"Overlay video saved to: {output_path}")