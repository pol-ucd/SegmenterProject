import argparse

import torch
from torch import GradScaler

from nn.data import data_load
from nn.models import SegformerBinarySegmentation, SegformerBinarySegmentation2
from nn.modules import CombinedLoss
from utils.torch_utils import TrainingManager, get_default_device


# TODO: this function needs to be reworked. Ignoring it for now.
# def compute_metrics(preds, targets, threshold=0.5):
#     preds = (preds > threshold).float()
#     targets = (targets > threshold).float()
#     dice_loss = DiceScore()
#     tp = (preds * targets).sum()
#     fp = (preds * (1 - targets)).sum()
#     fn = ((1 - preds) * targets).sum()
#     eps = 1e-7
#     return {'dice': dice_loss(preds, targets).item(),
#             # 'dice': (2 * tp / (2 * tp + fp + FN + eps)).item(),
#             'iou': (tp / (tp + fp + fn + eps)).item(),
#             'precision': (tp / (tp + fp + eps)).item(),
#             'recall': (tp / (tp + fn + eps)).item()}

def validate_positive_integer(value):
    """
    Custom type function for argparse to ensure an integer is greater than zero.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"'{value}' is an invalid positive integer value. Must be greater than zero.")
    return ivalue


def validate_01_float(value):
    """
    Custom type function for argparse to ensure a float is between 0 and 1.0
    """
    fvalue = float(value)
    if fvalue <= 0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError(f"'{value}' is an invalid float value. Must be in the range [0, 1].")
    return fvalue


def process_args():
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "--train_path",
        type=str,
        nargs='?',
        default="data/Polyp Segmentation/train",
        help="The path to the training data."
    )

    parser.add_argument(
        "--val_path",
        type=str,
        nargs='?',
        default="data/Polyp Segmentation/valid",
        help="The path to the validation data."
    )

    parser.add_argument(
        "--n_epochs",
        type=validate_positive_integer,
        nargs="?",  # Makes the argument optional
        default=4,  # Sets the default value if not provided
        help="The number of training epochs to run (default: 4)."
    )

    parser.add_argument(
        "--n_batch",
        type=validate_positive_integer,
        nargs="?",  # Makes the argument optional
        default=4,  # Sets the default value if not provided
        help="The number of records per mini batch for training and validation (default: 4)."
    )

    parser.add_argument(
        "--test_split",
        type=validate_01_float,
        nargs="?",  # Makes the argument optional
        default=0.3,  # Sets the default value if not provided
        help="The fraction of records to hold back for validation (default: 0.3)."
    )

    # Parse the arguments from the command line
    return parser.parse_args()


def main():
    args = process_args()

    print(args)

    # Set the default device to the best available GPU ... or CPU if no GPU available
    device = get_default_device()
    print(f"Using {device} device for model training.")

    """
    I've implemented a data_load function that
    can generate a train/test split if needed - but for now I'm just taking 100% 
    of the training and 100% validation data and using them to train and then to 
    validate respectively.
    """
    (train_loader,
     _) = data_load(args.train_path,
                    # test_split=args.test_split,
                    test_split=0.0,  # Use 100% for training
                    batch_size=args.n_batch,
                    verbose=True)

    (_,
     val_loader) = data_load(args.val_path,
                             # test_split=args.test_split,
                             test_split=1.0,  # Use 100% for testing/validation
                             batch_size=args.n_batch,
                             verbose=True)

    n_val = len(val_loader) * args.n_batch
    n_train = len(train_loader) * args.n_batch

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(val_loader)}")

    pretained_model = 'nvidia/segformer-b4-finetuned-ade-512-512'
    # model = SegformerBinarySegmentation().to(device)  #Old Word doc model
    model = SegformerBinarySegmentation2(pretrained_model=pretained_model).to(device)
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
                              save_preds=False,
                              save_preds_path=""
                              )
    train_params = {}
    eval_params = {}
    best_dice_score = 0.0
    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        train_loss, train_dice = trainer.train(**train_params)
        print(f"Train Loss: {train_loss / n_train:.4f}, Train Dice: {train_dice / n_train:.4f}")

        val_loss, val_metrics = trainer.evaluate(**eval_params)
        print(
            f"Total evaluation Loss: {val_loss / n_val:.4f} | Dice: {val_metrics['dice'] / n_val:.4f} | IOU: {val_metrics['iou'] / n_val:.4f}")
        if val_metrics['dice'] > best_dice_score:
            best_dice_score = val_metrics['dice']
            torch.save(model.state_dict(), "best_segformer.pth")
            # _, _ = trainer.evaluate(save_preds=True)
            print(f"Model saved for dice score: {val_metrics['dice'] / n_val:.4f}")


if __name__ == "__main__":

   main()