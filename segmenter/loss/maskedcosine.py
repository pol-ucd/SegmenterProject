# --------------------------------------------------------------------------
# 5) Masked Cosine Similarity Loss Function
# --------------------------------------------------------------------------
from typing import Union, Tuple, List, Optional

import torch
import torch.nn.functional as F

from segmenter.loss import BaseLoss
def masked_cosine_similarity_loss(predictions, targets, patch_mask, reduce='mean'):
    # assume predictions is a list of tensors (B, C, H, W)
    # keep everything as tensors on the correct device/dtype
    device = predictions[0].device
    dtype = predictions[0].dtype

    visible_mask = (1.0 - patch_mask).to(device=device, dtype=dtype)  # shape (B,1,H,W)
    total_visible_patches = visible_mask.sum()                       # tensor, not .item()

    # If no visible patches, return a zero that's connected to the graph by constructing it
    # from an input tensor so it has grad_fn when multiplied by 0.
    if total_visible_patches.item() == 0:
        return (predictions[0].sum() * 0.0).to(device=device, dtype=dtype)

    total_loss = torch.zeros((), dtype=dtype, device=device)

    for emb_A, emb_B in zip(predictions, targets):
        emb_A = emb_A.to(device=device, dtype=dtype)
        emb_B = emb_B.to(device=device, dtype=dtype)

        emb_A_norm = F.normalize(emb_A, p=2, dim=1)
        emb_B_norm = F.normalize(emb_B, p=2, dim=1)

        # per-patch cosine similarity (sum over channel dim to get scalar per spatial location)
        similarity = (emb_A_norm * emb_B_norm).sum(dim=1)  # shape (B,H,W)

        stage_loss = 1.0 - similarity                       # shape (B,H,W)
        masked_stage_loss = stage_loss * visible_mask.squeeze(1)  # shape (B,H,W)

        # denom as tensor so division stays in tensor world
        denom = total_visible_patches * emb_A.shape[1]
        total_loss = total_loss + masked_stage_loss.sum() / denom

    final_loss = total_loss / float(len(predictions))
    return final_loss


class MaskedCosineSimilarityLoss(BaseLoss):
    def __init__(self, reduce: Optional[str] = 'mean'):
        super(MaskedCosineSimilarityLoss, self).__init__()
        self.reduce = reduce

    def __call__(self, embeddings_A: Union[Tuple, List], embeddings_B: Union[Tuple, List],
                                  patch_mask: torch.Tensor, reduce: Optional[str] = None) -> torch.Tensor:
        self.reduce = reduce if reduce is not None else "mean"
        return masked_cosine_similarity_loss(predictions=embeddings_A,
                                             targets=embeddings_B,
                                             patch_mask=patch_mask, reduce=reduce)


    forward = __call__



if __name__ == '__main__':
    b, h, w = 8,512,512
    hidden_dims = [64, 128, 320, 512]
    embedding_a = [torch.randn(b, hd, h, w).clip(0, 1) for hd in hidden_dims]
    embedding_b = embedding_a
    mask = torch.ones(b, 1, h, w).float()
    mask[:, : , :h//2, :w//2] = 0.0

    loss_fn = MaskedCosineSimilarityLoss()

    loss = loss_fn(embedding_a, embedding_b, patch_mask=mask, reduce='mean')

    print(loss.item())

