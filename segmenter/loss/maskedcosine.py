# --------------------------------------------------------------------------
# 5) Masked Cosine Similarity Loss Function
# --------------------------------------------------------------------------
from typing import Union, Tuple, List, Optional

import torch
import torch.nn.functional as F

from segmenter.core import report_cuda_memory_usage
from segmenter.loss import BaseLoss


def masked_cosine_similarity_loss(predictions, targets, patch_mask):
    device = predictions[0].device
    dtype = predictions[0].dtype

    visible_mask = patch_mask  # (B,1,H,W)
    total_visible = visible_mask.sum().item()  # tensor

    # If no visible patches return zero that is attached to the graph
    if total_visible == 0:
        return (predictions[0].sum() * 0.0)

    total_loss = 0.0

    for emb_A, emb_B in zip(predictions, targets):
        emb_A = emb_A / emb_A.norm(dim=1, keepdim=True)
        emb_B = emb_B / emb_B.norm(dim=1, keepdim=True)
        similarity_matrix = torch.matmul(emb_A, emb_B.transpose(-2, -1))

        denom = total_visible * emb_A.shape[1]
        masked = (1.0 - similarity_matrix) * visible_mask  # shape (B,H,W)

        total_loss += masked.sum() / denom

    final_loss = total_loss / float(len(predictions))

    return final_loss


def enc_cosine_similarity_loss(predictions, targets):
    device = predictions[0].device
    dtype = predictions[0].dtype
    eps = 1e-08
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

    for emb_A, emb_B in zip(predictions, targets):
        emb_A = emb_A / (emb_A.norm(dim=1, keepdim=True) + eps)
        emb_B = emb_B / (emb_B.norm(dim=1, keepdim=True) + eps)
        similarity_matrix = torch.matmul(emb_A, emb_B.transpose(-2, -1))

        total_loss += 1.0 - similarity_matrix.mean()
    if not torch.isfinite(total_loss):
        print("Infinite ....>!!!")
    return total_loss


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
