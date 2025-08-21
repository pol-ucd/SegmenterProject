
import pandas as pd
import torch
from torch import GradScaler
from torch.utils.data import DataLoader

from nn.data import (split_images_and_masks,
                     SemanticSegmentationDatasetAugmentor,
                     SemanticSegmentationDatasetBasic)
from nn.models import SegformerBinarySegmentation4
from nn.modules import EarlyStopping, HybridLoss
from utils.torch_utils import RunManager


def main():

    test_split = 0.1
    num_classes = 2  # Binary classification - so masks and predictions will have shape [B, num_classes, H, W]
    batch_size = 4
    num_workers = 0
    n_augments = 2
    image_size = (512, 512)
    learning_rate = 1e-5 # Low learning rate for Segformer models
    l2_decay_penalty = 8e-4  # L2 regularization to prevent large weights
    n_epochs = 100
    stopper_patience = 5
    pretained_model = 'nvidia/segformer-b4-finetuned-ade-512-512'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_model_name = "best_dice_model.pth"

    print(f"Using {device} device for model training.")

    """
    Load the list of training and testing images and masks.
    We take the same percentage split from each separate source of images so
    that we are guaranteed to have a representation from each source in the
    training and testing sets.
    """
    train_images, train_masks, val_images, val_masks = split_images_and_masks(split=test_split)

    """
    Save the names of the files in the validation/test subset for 
    later use
    """
    df_file = pd.DataFrame({"val_image": val_images,
                            "val_masks": val_masks, })
    df_file.to_csv("validate_files.csv",
                   index=False)


    """ 
    Data sets and loaders
    """
    """
    Only use the SemanticSegmentationDatasetAugmentor class for 
    training data sine it randomly augments the available data
    to create more training data - and so is not suitable for 
    test or validation
    """
    train_ds = SemanticSegmentationDatasetAugmentor(
        train_images,
        train_masks,
        n_augments=n_augments,
        image_size=image_size
    )

    """
    Use the SemanticSegmentationDatasetBasic class for 
    validation or test. It does not perform any augmentations
    other than resizing and standard normalisation of the 
    images. Masks are not normalised.
    """
    val_ds = SemanticSegmentationDatasetBasic(
        val_images,
        val_masks,
        image_size=image_size
    )



    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(val_loader)}")

    """
    Setup the model 
    
    """

    model = SegformerBinarySegmentation4(pretrained_model=pretained_model,
                                         num_classes=num_classes)  #[B, num_classes, H, W]
    model.to(device)


    # cl_weights = {'bce': 0.5, 'tversky': 0.0, 'focal': 0.25, 'dice': 0.25, 'jaccard': 0.0}
    # loss_fn = CombinedLoss(weights=cl_weights)

    loss_fn = HybridLoss(weight_ce=0.5/1.9,
                         weight_dice=0.5/1.9,
                         weight_focal=0.2/1.9,
                         weight_tversky=0.2/1.9,
                         weight_iou=0.5/1.9,)

    # Initial freeze all parameters of the model
    print("Freezing encoder layers...")
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Unfreeze the decoder head and segmentation head
    print("Unfreezing decoder and segmentation head...")
    for param in model.base_model.decode_head.parameters():
        param.requires_grad = True

    # Only pass the parameters that require gradients to the optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=l2_decay_penalty # L2 regularization to prevent large weights
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, T_max = 50)



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

    early_stopper = EarlyStopping(patience=stopper_patience, min_delta=0.0001,
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
            f"Training Losses  : | Compound: {train_loss:.4f} | Dice: {train_dice:.4f} | IOU: {train_miou:.4f}")
        print(
            f"Evaluation Losses: | Compound: {val_loss:.4f} | Dice: {val_dice:.4f} | IOU: {val_miou:.4f}")
        print()

        scheduler.step(epoch + 1)

        early_stopper(val_miou, model, epoch)

        if early_stopper.early_stop:
            print(f"Training stopped early at epoch {epoch}")
            break

if __name__ == "__main__":
   main()