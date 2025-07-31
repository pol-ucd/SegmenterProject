import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss
from torch import GradScaler, autocast
from tqdm import tqdm

from utils.torch import get_default_device
from nn.data import data_load
from nn.models import SegformerBinarySegmentation

""" 
Some constants
"""
TEST_SPLIT = 0.3  # Passed to data_load to set the size of test/validation set
NUM_EPOCHS = 200
BEST_DICE = 0


def combined_loss(preds, targets):
    targets = targets.squeeze()
    preds = preds.squeeze()

    if preds.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(targets, size=preds.shape[-2:], mode='nearest')

    return (0.5 * bce(preds, targets)
            + 0.3 * tversky(torch.sigmoid(preds), targets)
            + 0.2 * focal(torch.sigmoid(preds),targets))

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
        with autocast(device_type='mps'):
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


if __name__ == "__main__":

    (train_loader,
     val_loader) = data_load(test_split=TEST_SPLIT)

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(val_loader)}")

    #Set the default device to the best available GPU ... or CPU if no GPU available
    device = get_default_device()
    # set_default_device(device)
    # device='cpu'

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

    #TODO: save heatmaps for visual comparison later
    # model.eval()
    # target_layer = model.decode_head[0]
    # grad_cam = GradCAM(model, target_layer)
    #
    # for images, masks in test_loader:
    #     images = images.to(device)
    #     for i in range(images.size(0)):
    #         input_img = images[i].unsqueeze(0)
    #         heatmap = grad_cam(input_img)
    #         show_gradcam_on_image(input_img[0], heatmap)
    #     # TODO: can we do this without a break statement?
    #     break  # only one batch
