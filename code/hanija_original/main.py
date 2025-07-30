import json
from glob import glob

import numpy as np

import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from sklearn.model_selection import train_test_split
from torch import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import SegformerModel
from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss


class PolypDataset(Dataset):
    def init(self, image_paths, mask_paths, transforms=None):
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

train_transform = A.Compose([ A.Resize(512, 512), A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.4), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5), A.GaussianBlur(p=0.2), A.Normalize(), ToTensorV2() ])
valid_transform = A.Compose([ A.Resize(512, 512), A.Normalize(), ToTensorV2() ])

train_imgs = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/train/.jpg'))
train_jsons = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/train/.json'))
valid_imgs = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/valid/.jpg'))
valid_jsons = sorted(glob('/kaggle/input/conor-data/Polyp Segmentation.v2i.sam2/valid/.json'))

all_imgs = train_imgs + valid_imgs
all_jsons = train_jsons + valid_jsons

(train_imgs, temp_imgs,
 train_jsons, temp_jsons) = train_test_split(all_imgs, all_jsons,
                                             test_size=0.3,
                                             random_state=42)
(val_imgs, test_imgs,
 val_jsons, test_jsons) = train_test_split(temp_imgs, temp_jsons,
                                           test_size=1/3,
                                           random_state=42)

train_ds = PolypDataset(train_imgs, train_jsons, train_transform)
val_ds = PolypDataset(val_imgs, val_jsons, valid_transform)
test_ds = PolypDataset(test_imgs, test_jsons, valid_transform)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

class SegformerBinarySegmentation(nn.Module):
    def init(self):
        super().init()
        self.backbone = SegformerModel.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        hidden_dim = self.backbone.config.hidden_sizes[-1]
        self.decode_head = nn.Sequential( nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
         nn.ReLU(inplace=True),
         nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, pixel_values):
        features = self.backbone(pixel_values=pixel_values).last_hidden_state  # [B, C, H/32, W/32]
        logits = self.decode_head(features)  # [B, 1, H/32, W/32]
        logits = F.interpolate(logits, size=pixel_values.shape[2:], mode='bilinear', align_corners=False)
        return logits  # [B, 1, 512, 512]

# TODO: Fix this to select device for more platform - including Apple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegformerBinarySegmentation().to(device)
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

def compute_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    TP = (preds * targets).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()
    eps = 1e-7
    return { 'dice': (2 * TP / (2 * TP + FP + FN + eps)).item(), 'iou': (TP / (TP + FP + FN + eps)).item(), 'precision': (TP / (TP + FP + eps)).item(), 'recall': (TP / (TP + FN + eps)).item() }

def train_one_epoch(loader):
    model.train()
    total_loss = 0
    total_dice = 0
    for images, masks in tqdm(loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(pixel_values=images)
            loss = combined_loss(logits, masks)
            probs = torch.sigmoid(logits)
            dice = compute_metrics(probs, masks)['dice']

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    total_loss += loss.item()
    total_dice += dice
    return total_loss / len(loader), total_dice / len(loader)

def evaluate(loader):
    model.eval()
    total_loss = 0
    total_metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(pixel_values=images)

    loss = combined_loss(logits, masks)
    probs = torch.sigmoid(logits)
    metrics = compute_metrics(probs, masks)
    total_loss += loss.item()
    for k in total_metrics:
        total_metrics[k] += metrics[k]
        avg_loss = total_loss / len(loader)
        avg_metrics = {k: total_metrics[k] / len(loader) for k in total_metrics}
        print(f"Val Loss: {avg_loss:.4f} | Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f}")
        return avg_loss, avg_metrics


class GradCAM:
    def init(self, model, target_layer):
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

        pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))  # (C,)
        activations = self.activations[0]  # (C, H, W)

        for i in range(pooled_gradients.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.sum(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        heatmap = cv2.resize(heatmap.numpy(), (inputs.size(3), inputs.size(2)))

        return heatmap

def show_gradcam_on_image(img_tensor, heatmap):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap),
                                cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * img + 0.5 * heatmap
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(overlay)
    plt.show()

if __name__=="__main__":
    num_epochs = 200
    best_dice = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_dice = train_one_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        val_loss, val_metrics = evaluate(val_loader)
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
        torch.save(model.state_dict(), "best_segformer.pth")
        print("Model saved!")

    model.eval()
    target_layer = model.decode_head[0]
    grad_cam = GradCAM(model, target_layer)


    for images, masks in test_loader:
        images = images.to(device)
        for i in range(images.size(0)):
            input_img = images[i].unsqueeze(0)
            heatmap = grad_cam(input_img)
            show_gradcam_on_image(input_img[0], heatmap)
        # TODO: can we do this without a break statement?
        break # only one batch

