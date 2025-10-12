from typing import Optional

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F, CrossEntropyLoss

from segmenter.models.msn import MSNSegFormerBase


class MSNBaseLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', symmetric: bool = False, eps: float = 1e-09):
        super().__init__()
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        else:
            self.reduction = None

        self.symmetric = symmetric
        self.eps = max(0, eps)
        self.device = None

    def forward(self, online_preds: torch.Tensor, target_protos: torch.Tensor) -> torch.Tensor:
        pass

    def _check_inputs(self, online_preds: torch.Tensor, target_protos: torch.Tensor):
        if online_preds.ndim != 2 or target_protos.ndim != 2:
            raise ValueError("online_preds and target_protos must be 2D tensors")

        if target_protos.shape[1] != online_preds.shape[1]:
            raise ValueError(f"Dim mismatch: online D={online_preds.shape[1]}, target D={target_protos.shape[1]}")


class MSNLoss(MSNBaseLoss):
    """
    MSN-style loss suitable for use with MoCoSiameseNetwork using
    patch-level embeddings from the SegFormer wrapper.

    Forward signature:
      loss = loss_fn(online_preds, target_protos)
    where:
      - online_preds: tensor (N_masked, D)  -- L2-normalized online projected embeddings (masked)
      - target_protos: tensor (N_all, D)     -- L2-normalized target projected embeddings (all patches)
    The loss computes a soft target distribution from the full target_protos similarity
    matrix (centered) and minimizes cross-entropy between that distribution and the
    online prediction distribution.
    The module maintains an exponential-moving-average center vector (`target_center`)
    which is subtracted from target_protos before building target distributions.

    Hyperparameters:
      - temperature: softmax temperature for logits
      - center_momentum: EMA momentum for center update (0..1)
    """

    def __init__(self, temperature: float = 0.1, center_momentum: float = 0.9,
                 eps: float = 1e-9, reduction='mean', symmetric=False):
        super().__init__(reduction, symmetric, eps)
        self.temperature = float(temperature)
        self.center_momentum = float(center_momentum)

        # center accumulates over target prototypes; will be created lazily on first update
        self.register_buffer("target_center", None, persistent=True)

    def _ensure_center(self, D: int, device: torch.device, dtype: torch.dtype):
        if getattr(self, "target_center", None) is None or self.target_center is None:
            # initialize center to zeros
            center = torch.zeros(D, device=device, dtype=dtype)
            # register buffer manually (already reserved name)
            object.__setattr__(self, "target_center", center)

    def update_center(self, target_protos: torch.Tensor):
        """
        Exponentially-smoothed update of the target center.
        Pass *normalized* target_protos (N_all, D) here, typically after encoder forward.
        This should be called after optimizer.step() and after EMA momentum encoder update.
        """
        if target_protos.numel() == 0:
            return
        D = target_protos.shape[1]
        self._ensure_center(D, target_protos.device, target_protos.dtype)

        batch_mean = target_protos.mean(dim=0)  # (D,)
        # ensure float32 accumulation for stability if needed
        bm = batch_mean.to(dtype=self.target_center.dtype)
        self.target_center.mul_(self.center_momentum).add_(bm * (1.0 - self.center_momentum))

    def forward(self, online_preds: torch.Tensor, target_protos: torch.Tensor) -> torch.Tensor:
        """
        Compute MSN loss.

        Steps:
          1. Ensure inputs are 2D and normalized.
          2. Center the target prototypes: target_protos - target_center.
          3. Compute similarities:
               logits_online = online_preds @ centered_targets.T  -> shape (N_masked, N_all)
               logits_target = centered_targets @ centered_targets.T -> shape (N_all, N_all)
          4. Build a soft target distribution from logits_target (row-wise softmax).
          5. Build predicted distribution from logits_online (row-wise softmax).
          6. Compute cross-entropy loss: -sum(target_dist * log(pred_dist)) averaged over online rows.
        """
        self._check_inputs(online_preds, target_protos)

        online_preds = F.normalize(online_preds, dim=-1)
        target_protos = F.normalize(target_protos, dim=-1)

        # defensive dtype/device handling
        device = online_preds.device
        dtype = online_preds.dtype

        N_masked, D = online_preds.shape
        # N_all = target_protos.shape[0]
        if target_protos.shape[1] != D:
            raise ValueError(f"Dim mismatch: online D={D}, target D={target_protos.shape[1]}")

        # initialize center if needed
        self._ensure_center(D, device, dtype)

        # center targets (use the buffer dtype)
        center = self.target_center.to(device=device, dtype=dtype)
        centered_targets = target_protos - center.unsqueeze(0)  # (N_all, D)

        # compute similarities
        # logits for online -> target (N_masked, N_all)
        logits_online = torch.matmul(online_preds, centered_targets.t()) / (self.temperature + self.eps)
        # predicted probabilities from online predictions (over columns = prototypes)
        logits_online = logits_online - logits_online.max(dim=1, keepdim=True)[0]
        pred_probs = F.softmax(logits_online, dim=1)  # (N_masked, N_all)

        # logits among targets to form soft targets (N_all, N_all)
        with torch.no_grad():
            logits_target = torch.matmul(centered_targets, centered_targets.t()) / (self.temperature + self.eps)
            # subtract max per row for numerical stability before softmax
            logits_target = logits_target - logits_target.max(dim=1, keepdim=True)[0]
            target_probs = F.softmax(logits_target, dim=1)  # (N_all, N_all)
            # Optional: apply constraints here

        # Build aggregated target distribution that matches online rows.
        # For each online sample we don't necessarily have a one-to-one mapping to a target row.
        # Simpler approach: average target_probs across rows to get a global prototype prior,
        # then use that as soft labels for online rows.
        #
        # Use the mean target distribution as soft labels to stabilize training.
        # target_distribution = target_probs.mean(dim=0, keepdim=True)  # (1, N_all)

        # target_distribution = target_distribution.expand(N_masked, -1)  # (N_masked, N_all)
        target_distribution = target_probs.expand(N_masked, -1)  # (N_masked, N_all)

        # KL Divergence between target_distribution (soft) and pred_probs
        # loss per online sample: -sum(target_dist * log(pred_probs/target_dist))
        entropy_target = - target_distribution * torch.log(target_distribution)
        loss_matrix = - target_distribution * torch.log(pred_probs) - entropy_target + self.eps
        loss = loss_matrix.sum(dim=1).mean()  # average over N_masked
        return loss


class SimSiamLoss(nn.Module):
    """
    Calculates the full symmetric SimSiam loss using MSE between predictions and targets.
    L_sym = 0.5 * [ L(p1, z2_det) + L(p2, z1_det) ]
    """

    def __init__(self):
        super().__init__()
        # MSE is used for the loss calculation
        self.mse_loss = nn.MSELoss()

    def forward(self, p1: torch.Tensor, z2_detached: torch.Tensor,
                p2: torch.Tensor, z1_detached: torch.Tensor) -> torch.Tensor:
        """
        Calculates the full symmetric loss.
        :param p1: Prediction from view 1 (Anchor) [B, D].
        :param z2_detached: Target embedding from detached view 2 (Positive) [B, D].
        :param p2: Prediction from view 2 (Positive) [B, D].
        :param z1_detached: Target embedding from detached view 1 (Anchor) [B, D].
        :return: Scalar total symmetric loss tensor.
        """
        B = p1.shape[0]
        # Normalize embeddings before calculating loss (as per SimSiam implementation)
        p1 = F.normalize(p1, dim=1)
        z2_detached = F.normalize(z2_detached, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1_detached = F.normalize(z1_detached, dim=1)

        # Calculate the two symmetric loss terms (MSE)
        # Term 1: Prediction from view 1 vs Target from detached view 2
        loss1 = self.mse_loss(p1, z2_detached)

        # Prediction from view 2 vs Target from detached view 1
        loss2 = self.mse_loss(p2, z1_detached)

        # 3. Total Symmetric Loss (Averaged)
        total_loss = 0.5 * (loss1 + loss2)

        return total_loss / B


#
# class InfoNCELoss(MSNBaseLoss):
#     """
#     SimCLR-style InfoNCE Loss for contrastive learning.
#     Calculates the loss over the full 2B embeddings (Anchor and Positive views).
#     """
#
#     def __init__(self, temperature=0.1, reduction='mean', symmetric=False):
#         super().__init__(reduction, symmetric)
#         self.temperature = temperature
#
#
#     def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
#         """
#         :param z_anchor: Anchor embeddings [N, B].
#         :param z_positive: Positive embeddings [N, B].
#         :return: Scalar InfoNCE loss.
#         """
#         self._check_inputs(z_anchor, z_positive)
#
#         N, B = z_anchor.shape
#
#         # Normalize embeddings (Crucial for cosine similarity)
#         z_anchor = F.normalize(z_anchor, dim=1)
#         z_positive = F.normalize(z_positive, dim=1)
#
#         similarity_matrix = torch.einsum('i c h w, j c h w -> i j c h w',
#                                          z_anchor, z_positive) / self.temperature
#
#         logits = similarity_matrix.reshape(B * B, -1)
#
#         # The label for the positive pair is always 0 (it is in the 0th column)
#         labels = torch.eye(B, dtype=torch.long, device=logits.device).reshape(B * B)
#
#         # Apply Cross Entropy Loss (equivalent to InfoNCE)
#         loss = F.cross_entropy(logits, labels)
#
#         return loss / (B * B)
#
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.25, eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        return contrastive_loss(z_anchor, z_positive, temperature=self.temperature, eps=self.eps)


def contrastive_loss(z1, z2, mask=None, temperature=0.1):
    """
    Computes contrastive loss between two batches of embeddings with optional masking and zero self-loss.

    Args:
        z1 (Tensor): Embeddings from view 1 (batch_size x dim)
        z2 (Tensor): Embeddings from view 2 (batch_size x dim)
        mask (Tensor, optional): Binary mask (batch_size,) or (batch_size x 1) indicating valid samples
        temperature (float): Temperature scaling factor

    Returns:
        Tensor: Scalar contrastive loss
    """
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Similarity matrix
    sim_matrix = torch.mm(z1, z2.t()) / temperature  # shape: (batch_size, batch_size)

    # Mask out self-similarity
    self_mask = torch.eye(batch_size, device=z1.device).bool()
    sim_matrix.masked_fill_(self_mask, float('-inf'))

    # Targets: each row should match the corresponding column index
    targets = torch.arange(batch_size, device=z1.device)

    # Compute loss
    loss_1 = F.cross_entropy(sim_matrix, targets, reduction='none')
    loss_2 = F.cross_entropy(sim_matrix.t(), targets, reduction='none')
    loss = 0.5 * (loss_1 + loss_2)

    if mask is not None:
        mask = mask.float().view(-1)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return loss.mean()


def _check_2d(z: torch.Tensor, name: str):
    if z.ndim != 2:
        raise ValueError(f"{name} must be 2D tensor of shape (N, D), got shape {z.shape}")


def nt_xent_image_level(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Standard SimCLR NT-Xent (InfoNCE) loss for image-level embeddings.

    Args:
      z1, z2: (B, D) L2-normalized embeddings for two augmented views; must have same B.
      temperature: positive scalar.

    Returns:
      scalar loss averaged over 2B examples.
    """
    _check_2d(z1, "z1")
    _check_2d(z2, "z2")
    if z1.shape[0] != z2.shape[0]:
        raise ValueError("z1 and z2 must have same batch size")

    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # mask self-similarities
    diag_mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim_masked = sim.masked_fill(diag_mask, -float("inf"))

    # positives: i <-> i+B
    positives = torch.arange(B, device=sim.device)
    positives = torch.cat([positives + B, positives], dim=0)  # (2B,)

    log_prob = F.log_softmax(sim_masked, dim=1)
    loss = -log_prob[torch.arange(2 * B, device=sim.device), positives]
    return loss.mean()


def nt_xent_general(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5,
                    positive_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Generalized NT-Xent for possibly unequal counts (z1: N1 x D, z2: N2 x D).

    Args:
      z1, z2: 2D tensors of embeddings.
      temperature: scalar.
      positive_index: optional LongTensor of length N = N1+N2 mapping each row index i in
        concatenated tensor [z1; z2] -> index of its positive in the concatenated tensor.
        If None and N1==N2 the standard pairing i <-> i+N1 is used.

    Returns:
      scalar loss averaged over N rows.
    """
    _check_2d(z1, "z1")
    _check_2d(z2, "z2")
    N1, N2 = z1.shape[0], z2.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (N, D), N = N1+N2
    N = N1 + N2

    sim = torch.matmul(z, z.T) / temperature  # (N, N)
    diag_mask = torch.eye(N, device=sim.device, dtype=torch.bool)
    sim_masked = sim.masked_fill(diag_mask, -1e9)

    if positive_index is None:
        if N1 != N2:
            raise ValueError("positive_index required when z1 and z2 have different lengths")
        pos = torch.arange(N1, device=sim.device)
        positive_index = torch.cat([pos + N1, pos], dim=0)  # (N,)
    else:
        if positive_index.numel() != N:
            raise ValueError("positive_index length must equal total concatenated rows N1+N2")
        positive_index = positive_index.to(device=sim.device)

    log_prob = F.log_softmax(sim_masked, dim=1)
    loss = -log_prob[torch.arange(N, device=sim.device), positive_index]
    return loss.mean()


class NTXentLoss(nn.Module):
    """
    Module wrapper for NT-Xent.

    Usage:
      loss_fn = NTXentLoss(temperature=0.5)
      loss = loss_fn(z1, z2)           # image-level case
      loss = loss_fn(z1, z2, mapping)  # generalized case with positive_index
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                positive_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        # expect inputs to be L2-normalized; normalize defensively
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        if positive_index is None:
            a = nt_xent_image_level(z1, z2, self.temperature)
            b1 = nt_xent_image_level(z1, z1, self.temperature)
            b2 = nt_xent_image_level(z2, z2, self.temperature)
        else:
            a = nt_xent_general(z1, z2, self.temperature, positive_index)
            b1 = nt_xent_general(z1, z1, self.temperature, positive_index)
            b2 = nt_xent_general(z2, z2, self.temperature, positive_index)
        return a - 0.5 * (b1 + b2)


class NegCosineSimilarityLoss(MSNBaseLoss):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.cosine_similarity(z1, z2)

    @staticmethod
    def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        p: (B, D') predicted vectors (already normalized)
        z: (B, D) target vectors (should be normalized prior to use)
        Returns mean negative cosine similarity per batch.
        """
        z = F.normalize(z, dim=1)
        return - (p * z).sum(dim=1).mean()


def compute_loss_sim(self, p1: torch.Tensor, z2_det: torch.Tensor, p2: torch.Tensor,
                 z1_det: torch.Tensor) -> torch.Tensor:
    """
        Symmetric SimSiam loss:
          loss = 0.5 * (neg_cos(p1, z2_det) + neg_cos(p2, z1_det))
        """
    loss1 = NegCosineSimilarityLoss.negative_cosine_similarity(p1, z2_det)
    loss2 = NegCosineSimilarityLoss.negative_cosine_similarity(p2, z1_det)
    return 0.5 * (loss1 + loss2)


if __name__ == '__main__':
    temperature = 0.1
    loss_fn = NTXentLoss(temperature=temperature)
    data_shape = (81, 1280)

    ce_fn = CrossEntropyLoss()

    torch.manual_seed(0)
    for _ in range(100):
        z_anchor = torch.randint(-1000, 1000, data_shape).float()
        z_positive = torch.randint(-1000, 1000, data_shape).float()

        msn_loss = loss_fn(z_anchor, z_positive)
        msn_zero = loss_fn(z_anchor, z_anchor)
        con_loss = contrastive_loss(z_anchor, z_positive)
        con_zero = contrastive_loss(z_anchor, z_anchor)
        print(
            f"MSNLoss : {msn_loss: 0.6f}, z_anchor self loss: [{msn_zero: 0.6f}], contrastive_loss: {con_loss: 0.6f}, z_anchor self loss: [{con_zero: 0.6f}]")
