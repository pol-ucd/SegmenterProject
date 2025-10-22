# --------------------------------------------------------------------------
# 5) Masked Cosine Similarity Loss Function
# --------------------------------------------------------------------------
from typing import Union, Tuple, List, Optional

import torch
import torch.nn.functional as F

from segmenter.loss import BaseLoss

def masked_cosine_similarity_loss(predictions, targets, patch_mask, reduce='mean'):
    device = predictions[0].device
    dtype = predictions[0].dtype

    visible_mask = (1.0 - patch_mask).to(device=device, dtype=dtype)  # (B,1,H,W)
    total_visible = visible_mask.sum()                               # tensor

    # If no visible patches return zero that is attached to the graph
    if total_visible.item() == 0:
        return (predictions[0].sum() * 0.0).to(device=device, dtype=dtype)

    total_loss = torch.zeros((), dtype=dtype, device=device)

    for emb_A, emb_B in zip(predictions, targets):
        emb_A = emb_A.to(device=device, dtype=dtype)
        emb_B = emb_B.to(device=device, dtype=dtype)

        emb_A_n = F.normalize(emb_A, p=2, dim=1)
        emb_B_n = F.normalize(emb_B, p=2, dim=1)

        # per-patch cosine similarity reduced over channel dim
        similarity = (emb_A_n * emb_B_n).sum(dim=1)    # shape (B,H,W)
        print("similarity: ", similarity)
        stage_loss = 1.0 - similarity                   # shape (B,H,W)

        masked = stage_loss * visible_mask.squeeze(1)  # shape (B,H,W)
        denom = total_visible * emb_A.shape[1]         # tensor * int -> tensor

        total_loss = total_loss + masked.sum() / denom

    final_loss = total_loss / float(len(predictions))
    print(final_loss)
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

