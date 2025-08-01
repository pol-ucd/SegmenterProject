import torch
from torch import GradScaler

from utils.torch import TrainingManager, get_default_device
from nn.modules import CombinedLoss, DiceScore
from nn.data import data_load
from nn.models import SegformerBinarySegmentation

""" 
Some constants
"""
TEST_SPLIT = 0.3  # Passed to data_load to set the size of test/validation set
BATCH_SIZE = 8
NUM_EPOCHS = 1  # Training epochs
SAVE_PREDS_PATH = "data/Polyp Segmentation/predicted_masks"


# TODO: this function needs to be reworked. Ignoring it for now and just returning Dice Score
def compute_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    dice_loss = DiceScore()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    eps = 1e-7
    return {'dice': dice_loss(preds, targets).item(),
            # 'dice': (2 * tp / (2 * tp + fp + FN + eps)).item(),
            'iou': (tp / (tp + fp + fn + eps)).item(),
            'precision': (tp / (tp + fp + eps)).item(),
            'recall': (tp / (tp + fn + eps)).item()}

if __name__ == "__main__":

    (train_loader,
     val_loader,
     train_names,
     val_names) = data_load(test_split=TEST_SPLIT, batch_size=BATCH_SIZE)

    n_val = len(val_loader)*BATCH_SIZE
    n_train = len(train_loader)*BATCH_SIZE

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(val_loader)}")

    #Set the default device to the best available GPU ... or CPU if no GPU available
    device = get_default_device()
    print(f"Using {device} device for model training.")

    model = SegformerBinarySegmentation().to(device)
    loss_fn = CombinedLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    """
    Only use GradScaler if we have CUDA
    """
    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()

    trainer = TrainingManager(model,
                              optimizer,
                              criterion=loss_fn,
                              scaler=scaler,
                              train_loader=train_loader,
                              eval_loader=val_loader,
                              device=device
                              )
    best_dice_score = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss, train_dice = trainer.train()
        print(f"Train Loss: {train_loss/n_train:.4f}, Train Dice: {train_dice/n_train:.4f}")

        val_loss, val_metrics = trainer.evaluate(save_preds=False, save_preds_path="")
        print(f"Total evaluation Loss: {val_loss/n_val:.4f} | Dice: {val_metrics['dice']/n_val:.4f} | IOU: {val_metrics['iou']/n_val:.4f}")
        if val_metrics['dice'] > best_dice_score:
            best_dice_score = val_metrics['dice']
            torch.save(model.state_dict(), "best_segformer.pth")
            # _, _ = trainer.evaluate(save_preds=True)
            print(f"Model saved for dice score: {val_metrics['dice']:.4f}")

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
