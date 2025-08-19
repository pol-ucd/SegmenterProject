
import pandas as pd
import torch
from torch import GradScaler
from torch.utils.data import DataLoader

from nn.data import SegmentationDataset, split_images_and_masks
from nn.models import SegformerBinarySegmentation4
from nn.modules import EarlyStopping, CombinedLoss
from transforms.images import ValidationImageTransforms, TrainingImageTransforms
from utils.torch_utils import RunManager


def main():

    test_split = 0.1
    num_classes = 2  # Binary classification {'not_lesion': 0, 'lesion': 1}
    ignore_index = 255
    batch_size = 4
    num_workers = 0
    n_epochs = 100
    pretained_model = 'nvidia/segformer-b4-finetuned-ade-512-512'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_model_name = "best_dice_model.pth"

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


    print(f"Using {device} device for model training.")

    """
    I've implemented a data_load function that
    can generate a train/test split if needed - but for now I'm just taking 100% 
    of the training and 100% validation data and using them to train and then to 
    validate respectively.
    """

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

    n_val = len(val_loader)
    n_train = len(train_loader)

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(val_loader)}")

    model = SegformerBinarySegmentation4(pretrained_model=pretained_model,
                                         num_classes=1)
    model.to(device)

    # loss_fn = DL(mode='binary')
    cl_weights = {'bce': 0.1, 'tversky': 0.2, 'focal': 0.2, 'dice': 0.6, 'jaccard': 0.6}
    loss_fn = CombinedLoss(weights=cl_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    """
    Only use GradScaler if we have CUDA
    """
    scaler = None
    if torch.cuda.is_available():
        scaler = GradScaler()

    trainer = RunManager(model,
                         optimizer,
                         criterion=loss_fn,
                         scaler=scaler,
                         train_loader=train_loader,
                         eval_loader=val_loader,
                         save_preds=False,
                         save_preds_path=""
                         )
    train_params = {}
    eval_params = {}

    early_stopper = EarlyStopping(patience=10, min_delta=0.0001,
                                  mode='min', verbose=True,
                                  save_path=save_model_name)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print()
        train_metrics = trainer.train(**train_params)
        val_metrics = trainer.evaluate(**eval_params)

        train_loss = train_metrics['loss']
        train_miou = train_metrics['iou']
        train_dice = train_metrics['dice']
        val_loss = val_metrics['loss']
        val_miou = val_metrics['iou']
        val_dice = val_metrics['dice']

        print(
            f"Training Losses  : | Compound Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IOU: {train_miou:.4f}")
        print(
            f"Evaluation Losses: | Compound Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IOU: {val_miou:.4f}")
        print()

        scheduler.step(epoch + 1)

        early_stopper(val_miou, model, epoch)

        if early_stopper.early_stop:
            print(f"Training stopped early at epoch {epoch}")
            break

if __name__ == "__main__":
   main()