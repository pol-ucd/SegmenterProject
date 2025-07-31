import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss
from torch import GradScaler, autocast
from tqdm import tqdm

from nn.data import data_load
from nn.models import SegformerBinarySegmentation
from utils.torch import get_default_device, set_default_device

""" 
Some constants
"""
TEST_SPLIT = 0.3  # Passed to data_load to set the size of test/validation set
NUM_EPOCHS = 200
BEST_DICE = 0


def combined_loss(preds, targets):
    if preds.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(targets, size=preds.shape[-2:], mode='nearest')
    return 0.5 * bce(preds, targets) + 0.3 * tversky(torch.sigmoid(preds), targets) + 0.2 * focal(torch.sigmoid(preds),
                                                                                                  targets)


def compute_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    TP = (preds * targets).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()
    eps = 1e-7
    return {'dice': (2 * TP / (2 * TP + FP + FN + eps)).item(), 'iou': (TP / (TP + FP + FN + eps)).item(),
            'precision': (TP / (TP + FP + eps)).item(), 'recall': (TP / (TP + FN + eps)).item()}


def train_one_epoch(model, optimizer, scaler, loader):
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


def evaluate(model, loader):
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
        print(
            f"Val Loss: {avg_loss:.4f} | Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f}")
        return avg_loss, avg_metrics


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


if __name__ == "__main__":

    (train_loader,
     val_loader) = data_load(test_split=TEST_SPLIT)

    print(f"Train Images: {len(train_loader)}")
    print(f"Val Images: {len(val_loader)}")

    #Set the default device to the best available GPU ... or CPU if no GPU available
    device = get_default_device()
    set_default_device(device)

    model = SegformerBinarySegmentation().to(device)
    bce = nn.BCEWithLogitsLoss()
    tversky = TverskyLoss(mode='binary', alpha=0.2, beta=0.4)
    focal = FocalLoss(mode='binary')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss, train_dice = train_one_epoch(model, optimizer, scaler, train_loader)
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")

        val_loss, val_metrics = evaluate(model, val_loader)
        if val_metrics['dice'] > BEST_DICE:
            BEST_DICE = val_metrics['dice']
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
        break  # only one batch
