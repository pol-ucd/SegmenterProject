"""
Various Loss functions and related utilities.

Note. The Loss functions below have signatures that are consistent with the type and shape
of prediction and target tensors used in the processed pipeline.

Target masks are always one hot encoded to be of shape [B, C, H, W]
Predictions are logits of shape [B, C, H, W]

Where:
    B = the batch dimension.
    C = the number of classes.
    H, W = the image height and width.

Target Masks have type Torch.Long
Predictions have type Torch.Float

Loss is returned as a single mean loss per batch and class. If the calling function
needs per sample loss the returned loss should be multiplied by the batch size.

"""

# try:
#     import FastGeodis
#
#     is_geo_installed = True
# except ImportError:
#     is_geo_installed = False

import torch

from segmenter.loss.boundary import make_soft_boundary
from segmenter.loss.distance import DistanceTransform2D
from segmenter.loss.factory import LossFactory
from segmenter.loss.hybrid import HybridLoss
from segmenter.loss.utils import one_hot, sobel_grad

# ---------------------------
# Global Constants
# ---------------------------
# EPSILON = 1e-6


# ---------------------------
# Utilities
# ---------------------------


#
# class PairedSpatialAug(nn.Module):
#     def __init__(self,
#                  degrees: float = 10.0,
#                  translate: tuple = (0.1, 0.1),
#                  scale: tuple = (0.9, 1.1),
#                  p: float = 0.5):
#         super().__init__()
#         # random affine will be applied consistently across frames
#         self.affine = K.VideoSequential(
#             K.RandomAffine(
#                 degrees=degrees,
#                 translate=translate,
#                 scale=scale,
#                 p=p
#             ),
#             data_format="BCTHW",
#             same_on_frame=True
#         )
#
#     def forward(self, video: torch.Tensor, mask: torch.Tensor):
#         """
#         video: Tensor (B, 3, T, H, W)
#         mask:  Tensor (B, 1, T, H, W)
#         """
#         # 1. Concatenate image + mask channels
#         combined = torch.cat([video, mask], dim=1)  # (B, 4, T, H, W)
#
#         # 2. Apply spatial augmentation
#         augmented = self.affine(combined)           # still (B, 4, T, H, W)
#
#         # 3. Split them back
#         vid_aug  = augmented[:, :3]
#         mask_aug = augmented[:, 3:].round()         # round if mask is binary
#
#         return vid_aug, mask_aug
#

# class ImageLightingAugmentation(nn.Module):
#     def __init__(
#         self,
#         brightness: tuple = (0.7, 1.3),
#         contrast:   tuple = (0.7, 1.3),
#         saturation: tuple = (0.7, 1.3),
#         hue:        tuple = (-0.1, 0.1),
#         gamma:      tuple = (0.8, 1.2),
#         erase_scale:tuple = (0.02, 0.08),
#         p_jitter:   float = 0.5,
#         p_gamma:    float = 0.5,
#         p_shuffle:  float = 0.2,
#         p_erase:    float = 0.3
#     ):
#         """
#         Args:
#             brightness, contrast, saturation, hue: ranges for ColorJitter.
#             gamma: range for RandomGamma.
#             erase_scale: min/max area ratio for RandomErasing.
#             p_*: probability of applying each transform.
#         """
#         super().__init__()
#         # VideoSequential applies the same random parameters to all frames in a clip
#         self.augment = K.ImageSequential(
#             K.ColorJitter(brightness=brightness,
#                           contrast=contrast,
#                           saturation=saturation,
#                           hue=hue,
#                           p=p_jitter),
#             K.RandomGamma(gamma=gamma,
#                           p=p_gamma),
#             K.RandomChannelShuffle(p=p_shuffle),
#             K.RandomErasing(scale=erase_scale,
#                             p=p_erase)
#         )
#
#     def forward(self, video: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             video: Tensor of shape (B, C, T, H, W), values in [0,1] or [0,255].
#         Returns:
#             Augmented video tensor, same shape as input.
#         """
#         # If your video is in [0,255], convert to float in [0,1] first:
#         orig_dtype = video.dtype
#         if video.dtype != torch.float32:
#             video = video.float() / 255.0
#
#         # Apply lighting augmentations
#         augmented = self.augment(video)
#
#         # Convert back to original dtype/range
#         if orig_dtype != torch.float32:
#             augmented = (augmented * 255.0).to(orig_dtype)
#         return augmented



# ---------------------------
# Distance transform backends (GPU)
# ---------------------------


# ---------------------------
# Loss Components
# ---------------------------


if __name__ == '__main__':
    # ---------------------------
    # Test Calls for all functions and classes
    # ---------------------------

    n_test_epochs = 10

    # Setup for tests
    n_batch, h, w, n_classes = 3, 128, 128, 11
    print("\n--- Creating Individual Components ---")
    dice_loss_fn = LossFactory.create('dice')
    iou_loss_fn = LossFactory.create('iou')
    focal_loss_fn = LossFactory.create('focal', alpha=0.5, gamma=3.0)
    tversky_loss_fn = LossFactory.create('tversky', alpha=0.3, beta=0.7)
    ce_loss_fn = LossFactory.create('ce')
    boundary_sdf_loss_fn = LossFactory.create('boundary_sdf', dt_backend="kornia")
    soft_chamfer_loss_fn = LossFactory.create('soft_chamfer', dt_backend="kornia")

    loss_config_binary = {
        'bce': {'weight': 0.4},
        'ce': {'weight': 0.4},
        'tversky': {'weight': 0.6},
        'dice': {'weight': 0.5},
        'boundary_sdf': {'weight': 0.1, 'dt_backend': 'fastgeodis'}
    }
    hybrid_loss_binary = HybridLoss(loss_config_binary)

    loss_config_multi = {
        'ce': {'weight': 0.5},
        'dice': {'weight': 0.4},
        'focal': {'weight': 0.5, 'alpha': 1.0, 'gamma': 3.0},
        'boundary_sdf': {'weight': 0.1, 'dt_backend': 'kornia'}
    }
    hybrid_loss_multi = HybridLoss(loss_config_multi)

    dt_kornia = DistanceTransform2D(backend="kornia")
    dt_fastgeodis = DistanceTransform2D(backend="fastgeodis")

    print(f"Running {n_test_epochs} test epochs.")
    for n in range(n_test_epochs):
        print(f"Test epoch [{n+1} / {n_test_epochs}]")
        test_logits = torch.randn(n_batch, n_classes, h, w)
        test_mask = torch.randint(0, n_classes, (n_batch, h, w))
        test_mask = one_hot(test_mask, n_classes)
        binary_logits = torch.randn(n_batch, n_classes, h, w)
        binary_mask = torch.randint(0, n_classes, (n_batch, h, w))

        print("--- Testing Utility Functions ---")
        test_one_hot = one_hot(test_mask, n_classes)
        print(f"one_hot output shape: {test_one_hot.shape}")
        test_sobel_grad = sobel_grad(torch.randn(1, 1, h, w))
        print(f"sobel_grad output shape: {test_sobel_grad.shape}")
        test_soft_boundary = make_soft_boundary(torch.rand(1, 1, h, w))
        print(f"make_soft_boundary output shape: {test_soft_boundary.shape}")

        print("\n--- Testing DistanceTransform2D ---")
        # binary_map = (test_mask[:, None, :, :] == 1).float()
        binary_map = test_mask.float()
        print(f"DistanceTransform2D (kornia) EDT: {dt_kornia.edt(binary_map).mean().item():.4f}")
        print(f"DistanceTransform2D (fastgeodis) EDT: {dt_fastgeodis.edt(binary_map).mean().item():.4f}")
        print(f"DistanceTransform2D (kornia) SDT: {dt_kornia.signed_distance(binary_map).mean().item():.4f}")
        print(f"DistanceTransform2D (fastgeodis) SDT: {dt_fastgeodis.signed_distance(binary_map).mean().item():.4f}")


        print(f"Dice Loss: {dice_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"IoU Loss: {iou_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Focal Loss: {focal_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Tversky Loss: {tversky_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Cross-Entropy Loss: {ce_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Boundary SDF Loss: {boundary_sdf_loss_fn(test_logits, test_mask).item():.4f}")
        print(f"Soft Chamfer Loss: {soft_chamfer_loss_fn(test_logits, test_mask).item():.4f}")

        print("\n--- Testing HybridLoss Class ---")
        hybrid_total_loss_multi = hybrid_loss_multi(test_logits, test_mask)
        print(f"HybridLoss (multi-class) total loss: {hybrid_total_loss_multi.item():.4f}")

        hybrid_total_loss_binary = hybrid_loss_binary(binary_logits, binary_mask)
        print(f"HybridLoss (binary) total loss: {hybrid_total_loss_binary.item():.4f}")

    # ------------------------------
    """ Test for Perfect matches """
    # ------------------------------
    perfect_pred = torch.zeros(n_batch, n_classes, h, w)
    perfect_target = torch.zeros(n_batch, n_classes, h, w).long()
    perfect_pred[:n_batch, 0, :, :] = 100.0  # Set logits for class 0 to be very high
    perfect_target[:n_batch] = 0  # Ground truth is class 0 everywhere

    print("\n--- Testing DiceLoss with a perfect prediction ---")
    perfect_loss = dice_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "DiceLoss for perfect prediction is not near zero!"

    print("\n--- Testing IoULoss with a perfect prediction ---")
    perfect_loss = iou_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "IoULoss for perfect prediction is not near zero!"

    print("\n--- Testing FocalLoss with a perfect prediction ---")
    perfect_loss = focal_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "FocalLoss for perfect prediction is not near zero!"

    print("\n--- Testing TverskyLoss with a perfect prediction ---")
    perfect_loss = tversky_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "TverskyLoss for perfect prediction is not near zero!"

    print("\n--- Testing CrossEntropyLoss with a perfect prediction ---")
    perfect_loss = ce_loss_fn(perfect_pred,
                              perfect_target.float()).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "CrossEntropyLoss for perfect prediction is not near zero!"

    # Boundary SDF Loss
    print("\n--- Testing BoundarySDFLoss with perfect prediction ---")
    perfect_sdf_loss = boundary_sdf_loss_fn(perfect_pred, perfect_target).item()
    print(f"Boundary SDF Loss for perfect prediction: {perfect_sdf_loss:.6f}")
    assert perfect_sdf_loss < 1e-5, "Boundary SDF Loss for perfect prediction is not near zero!"

    print("\n--- Testing SoftChamferLoss with a perfect prediction ---")
    perfect_loss = soft_chamfer_loss_fn(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "SoftChamferLoss for perfect prediction is not near zero!"

    print("\n--- Testing HybridLoss with a perfect prediction ---")
    perfect_loss = hybrid_loss_multi(perfect_pred, perfect_target).item()
    print(f"Loss for perfect prediction: {perfect_loss:.6f}")
    assert perfect_loss < 1e-5, "SoftChamferLoss for perfect prediction is not near zero!"


