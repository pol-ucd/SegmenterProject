# --------------------------------------------------------------------------
# 5) Masked Cosine Similarity Loss Function
# --------------------------------------------------------------------------
from typing import Union, Tuple, List, Optional

import torch
import torch.nn.functional as F

from segmenter.core import report_cuda_memory_usage
from segmenter.loss import BaseLoss

def thresholded_downscale_mask(mask, downscale_factor, threshold=0.5):
    """
    Downscales a binary mask (B, 1, H, W) using an average pooling
    criterion and a subsequent threshold.

    Args:
        mask (torch.Tensor): The input binary mask (B, 1, H, W).
        downscale_factor (int): The factor by which to reduce H and W (e.g., 4 for 512->128).
        threshold (float): The percentage of '1's required in the patch (e.g., 0.5).

    Returns:
        torch.Tensor: The downscaled binary mask (B, 1, H/R, W/R).
    """
    # 1. Use Average Pooling to calculate the mean of each patch (4x4, 8x8, etc.)
    # The output tensor will contain a value between 0.0 and 1.0,
    # representing the density of '1's in the original patch.
    # Note: kernel_size and stride are set to the downscale factor (R).
    mean_pooled_mask = F.avg_pool2d(
        input=mask.float(),  # Must convert to float for mean calculation
        kernel_size=downscale_factor,
        stride=downscale_factor
    )

    # 2. Apply the threshold criterion
    # This binarizes the downscaled image: True (1) if mean > threshold, False (0) otherwise.
    downscaled_mask = (mean_pooled_mask > threshold).float()

    return downscaled_mask

def masked_cosine_similarity_loss(predictions, targets, mask):
    device = predictions[0].device
    dtype = predictions[0].dtype

    visible_mask = mask  # (B,1,H,W)
    total_visible = visible_mask.sum().item()  # tensor

    # If no visible patches return zero that is attached to the graph
    if total_visible == 0:
        return (predictions[0].sum() * 0.0)

    total_loss = 0.0

    for emb_A, emb_B in zip(predictions, targets):
        downscale_factor = int( mask.shape[-2] // emb_A.shape[-2])
        scaled_mask = thresholded_downscale_mask(mask, downscale_factor, threshold=0.5)
        total_visible = scaled_mask.sum().item()
        similarity = F.cosine_similarity(emb_A*scaled_mask,
                                         emb_B*scaled_mask,
                                         dim=1)

        denom = total_visible * emb_A.shape[1]

        total_loss += (1.0 - similarity).sum() / denom

    final_loss = total_loss / float(len(predictions))

    return final_loss

def enc_cosine_similarity_loss(predictions, targets):
    if isinstance(predictions, (list, tuple)):
        device = predictions[0].device
        dtype = predictions[0].dtype
        eps = 1e-08
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for emb_A, emb_B in zip(predictions, targets):
            similarity = F.cosine_similarity(emb_A, emb_B, dim=1)

            total_loss += 1.0 - similarity.mean()
        return total_loss
    else:
        similarity = F.cosine_similarity(predictions, targets, dim=1)
        return (1.0 - similarity).mean()


class MaskedCosineSimilarityLoss(BaseLoss):
    def __init__(self):
        super(MaskedCosineSimilarityLoss, self).__init__()

    def __call__(self, embeddings_A: Union[Tuple, List], embeddings_B: Union[Tuple, List],
                 mask: torch.Tensor, reduce: Optional[str] = None) -> torch.Tensor:
        self.reduce = reduce if reduce is not None else "mean"
        return masked_cosine_similarity_loss(predictions=embeddings_A,
                                             targets=embeddings_B,
                                             mask=mask)

    forward = __call__


class EncodingCosineSimilarityLoss(BaseLoss):
    def __init__(self):
        super(EncodingCosineSimilarityLoss, self).__init__()

    def __call__(self, embeddings_A: Union[Tuple, List],
                 embeddings_B: Union[Tuple, List]) -> torch.Tensor:
        return enc_cosine_similarity_loss(predictions=embeddings_A,
                                          targets=embeddings_B)

    forward = __call__


if __name__ == '__main__':
    b, h, w = 8, 512, 512
    hidden_dims = [64, 128, 320, 512]
    embedding_a = [torch.randn(b, hd, h, w).clip(0, 1) for hd in hidden_dims]
    embedding_b = embedding_a
    mask = torch.ones(b, 1, h, w).float()
    mask[:, :, :h // 2, :w // 2] = 0.0

    loss_fn = MaskedCosineSimilarityLoss()

    loss = loss_fn(embedding_a, embedding_b, patch_mask=mask, reduce='mean')

    print(loss.item())
