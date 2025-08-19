
import pandas as pd
import torch
from torch.utils.data import DataLoader

from losses import (DiceLoss as DL)
from nn.data import SegmentationDataset
from nn.models import SegformerBinarySegmentation4
from transforms.images import ValidationImageTransforms
from utils.torch_utils import RunManager


def main():

    num_classes = 2  # Binary classification {'not_lesion': 0, 'lesion': 1}
    ignore_index = 255
    batch_size = 4
    num_workers = 0
    n_epochs = 100
    pretained_model = 'nvidia/segformer-b4-finetuned-ade-512-512'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    Load the list of training and testing images and masks.
    We take the same percentage split from each separate source of images so
    that we are guaranteed to have a representation from each source in the
    training and testing sets.
    """


    df_files = pd.read_csv("validate_files.csv")

    print(f"Using {device} device for model training.")
    val_images = df_files.val_image.values
    val_masks = df_files.val_masks.values

    val_ds = SegmentationDataset(
        val_images, val_masks,
        transform=ValidationImageTransforms(size=(512, 512)),
        num_classes=num_classes, ignore_index=ignore_index
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)


    print(f"Validation batches: {len(val_loader)}")

    model = SegformerBinarySegmentation4(pretrained_model=pretained_model,
                                         num_classes=1)
    model.load_state_dict(torch.load("best_dice_model.pth", map_location=device))
    model.to(device)

    loss_fn = DL(mode='binary')

    trainer = RunManager(model,
                         optimizer=None,
                         criterion=loss_fn,
                         scaler=None,
                         train_loader=None,
                         eval_loader=val_loader,
                         save_preds=False,
                         save_preds_path=""
                         )

    eval_params = {}

    val_metrics = trainer.evaluate(**eval_params)

    val_loss = val_metrics['loss']
    val_miou = val_metrics['iou']
    val_dice = val_metrics['dice']

    print(
        f"Evaluation Losses: | Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IOU: {val_miou:.4f}")
    print()




if __name__ == "__main__":
   main()