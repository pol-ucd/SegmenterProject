import torch
from torch import nn as nn
from torch.nn import functional as F

#
# class MSNLoss(nn.Module):
#     """
#     Implements the Mean-Shifted Network (MSN) objective, minimizing
#     KL divergence between online predictions (Q) and centered target assignments (P).
#     """
#
#     def __init__(self, temperature: float = 0.1, center_momentum: float = 0.9):
#         super().__init__()
#         # 'temperature' controls the sharpness of the target distribution (P).
#         self.temperature = temperature
#
#         # Center Momentum (m): Controls the speed of the EMA update.
#         # MSN often uses m close to 0 (e.g., 0.1), while MoCo/DINO use m close to 1.
#         self.center_momentum = center_momentum
#
#         # Initialize target_center as a persistent buffer, not a trainable parameter.
#         self.register_buffer("target_center", None)
#
#     @torch.no_grad()
#     def update_center(self, target_protos_current_batch: torch.Tensor):
#         """
#         Updates the target center using an Exponential Moving Average (EMA).
#         This method must be called once per iteration on the target network's outputs.
#         """
#         # Ensure target_protos are detached (they should be, but safety check) and normalized
#         target_protos = F.normalize(target_protos_current_batch.detach(), dim=1)
#
#         # Calculate the mean of the current batch's target prototypes
#         current_batch_center = target_protos.mean(dim=0, keepdim=True)
#
#         # Initialize the center if it's the first run
#         if self.target_center is None:
#             self.target_center = current_batch_center.clone()
#             return
#
#         # EMA Update: c_new = (m) * c_old + (1 - m) * c_batch_mean
#         new_center = self.target_center.clone() * self.center_momentum + \
#                      current_batch_center * (1.0 - self.center_momentum)
#
#         # Copy the updated value back to the registered buffer
#         self.target_center.copy_(new_center)
#
#     def forward(self, online_preds: torch.Tensor, target_protos: torch.Tensor) -> torch.Tensor:
#         """
#         Calculates the KL divergence between the online network's predictions (Q)
#         and the target network's assignments (P).
#
#         :param online_preds: Predictions from the online network for MASKED patches [N_masked, D].
#         :param target_protos: Representations from the target network for ALL patches [N_all, D].
#         :return: Scalar loss tensor (mean KL divergence).
#         """
#         B = online_preds.shape[0]
#         # Normalization
#         online_preds = F.normalize(online_preds, dim=1)
#         target_protos = F.normalize(target_protos, dim=1)
#
#         # 2. Mean-Shift / Centering (The critical MSN step)
#         # If center is not initialized, update_center should be called first,
#         # but we use a robust check here.
#         if self.target_center is None:
#             # If the center hasn't been initialized, use the current batch mean as a proxy
#             centered_target_protos = target_protos - target_protos.mean(dim=0, keepdim=True)
#         else:
#             # Subtract the stabilized moving average center from all target prototypes
#             centered_target_protos = target_protos - self.target_center
#
#         # Similarity Matrix: Sim(Online_Masked, Centered_Target_All)
#         # Shape: (N_masked, N_all)
#
#         similarity_matrix = torch.matmul(online_preds,
#                                          centered_target_protos.transpose(-1, -2))
#
#         # Target Distribution (P) - Sharpened Softmax
#         # P = softmax(Sim / temperature). This is the 'teacher' signal.
#         targets = F.softmax(similarity_matrix / self.temperature, dim=1)
#
#         # Prediction Distribution (log Q) - Log Softmax
#         # log Q = log(softmax(Sim)). This is the 'student' prediction.
#         predictions = F.log_softmax(similarity_matrix, dim=1)
#
#         # KL Divergence / Cross-Entropy
#         # L = - sum(P * log Q) -> Minimizes KL(P || Q).
#         loss = - (targets * predictions).sum(dim=1)
#
#         return loss.mean()


class MSNLoss(nn.Module):
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

    def __init__(self, temperature: float = 0.1, center_momentum: float = 0.9, eps: float = 1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.center_momentum = float(center_momentum)
        self.eps = float(eps)

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
        if online_preds.ndim != 2 or target_protos.ndim != 2:
            raise ValueError("online_preds and target_protos must be 2D tensors")

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

        # logits among targets to form soft targets (N_all, N_all)
        with torch.no_grad():
            logits_target = torch.matmul(centered_targets, centered_targets.t()) / (self.temperature + self.eps)
            # subtract max per row for numerical stability before softmax
            logits_target = logits_target - logits_target.max(dim=1, keepdim=True)[0]
            target_probs = F.softmax(logits_target, dim=1)  # (N_all, N_all)
            # Optionally, one could sharpen or apply constraints here.

        # predicted probabilities from online predictions (over columns = prototypes)
        logits_online = logits_online - logits_online.max(dim=1, keepdim=True)[0]
        pred_probs = F.softmax(logits_online, dim=1)  # (N_masked, N_all)

        # Build aggregated target distribution that matches online rows.
        # For each online sample we don't necessarily have a one-to-one mapping to a target row.
        # Simpler approach: average target_probs across rows to get a global prototype prior,
        # then use that as soft labels for online rows.
        #
        # More faithful MSN variants pick the corresponding target prototype row for masked indices.
        # If masked positions correspond to some indices in target_protos, you can map them directly.
        #
        # Here we use the mean target distribution as soft labels to stabilize training.
        target_distribution = target_probs.mean(dim=0, keepdim=True)  # (1, N_all)
        target_distribution = target_distribution.expand(N_masked, -1)  # (N_masked, N_all)

        # cross-entropy between target_distribution (soft) and pred_probs
        # loss per online sample: -sum(target_dist * log(pred_probs))
        loss_matrix = - target_distribution * torch.log(pred_probs + self.eps)
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


class InfoNCELoss(nn.Module):
    """
    SimCLR-style InfoNCE Loss for contrastive learning.
    Calculates the loss over the full 2B embeddings (Anchor and Positive views).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        """
        :param z_anchor: Anchor embeddings [B, C(lasses), H, W].
        :param z_positive: Positive embeddings [B, C(lasses), H, W].
        :return: Scalar InfoNCE loss.
        """
        B, C, H, W = z_anchor.shape

        # Normalize embeddings (Crucial for cosine similarity)
        z_anchor = F.normalize(z_anchor, dim=1)
        z_positive = F.normalize(z_positive, dim=1)

        similarity_matrix = torch.einsum('i c h w, j c h w -> i j c h w',
                                         z_anchor, z_positive) / self.temperature

        logits = similarity_matrix.reshape(B * B, -1)

        # The label for the positive pair is always 0 (it is in the 0th column)
        labels = torch.eye(B, dtype=torch.long, device=logits.device).reshape(B * B)

        # Apply Cross Entropy Loss (equivalent to InfoNCE)
        loss = F.cross_entropy(logits, labels)

        return loss / (B * B)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.25, eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        return contrastive_loss(z_anchor, z_positive, temperature=self.temperature, eps=self.eps)

def contrastive_loss(out_1, out_2, temperature=0.25, eps=1e-06):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)

    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / (temperature + eps))
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / (temperature + eps))
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

class NTXEntLoss(nn.Module):
    def __init__(self, temperature:float=0.5, eps:float=1e-9):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2:torch.Tensor, temperature:float=None):
        # self.temperature = temperature if temperature is not None else self.temperature
        # B = z1.shape[0]
        # z = torch.cat([z1, z2], dim=0)
        # sim = torch.matmul(z, z.T) / self.temperature
        # sim = sim - torch.eye(2*B, device=sim.device) * self.eps
        # positives = torch.cat([torch.arange(B,2*B), torch.arange(0,B)], dim=0).to(z.device)
        # log_prob = sim.log_softmax(dim=1)
        # loss = -log_prob[torch.arange(2*B), positives].mean()
        # return loss
        return nt_xent_loss(z1, z2, self.temperature)

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Basic NT-Xent (SimCLR) loss for a batch of size B.
    z1, z2: (B, D) L2-normalized embeddings
    Returns scalar loss averaged over 2B samples.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)             # (2B, D)
    sim = torch.matmul(z, z.T) / temperature   # (2B, 2B)
    # mask out self similarities
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    # create mask to ignore same-sample similarities
    diag_mask = torch.eye(2 * B, device=z.device).bool()
    sim_masked = sim.masked_fill(diag_mask, -9e15)

    # positive pairs indices: i <-> i+B
    positives = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z.device)

    numerator = torch.exp(sim[torch.arange(2 * B), positives])
    denominator = torch.exp(sim_masked).sum(dim=1)
    loss = -torch.log(numerator / denominator)
    return loss.mean()



if __name__ == '__main__':
    temperature = 0.2
    N, D = 11, 256

    z_anchor = torch.randint(-255, 255, (N, D)).float()
    z_positive = torch.randint(-255, 255, (N, D)).float()

    loss_fn = MSNLoss(temperature=0.05, center_momentum=0.1)
    loss = loss_fn(z_anchor, z_positive)
    print(loss)

    loss = contrastive_loss(z_anchor, z_positive)
    print(loss)

    loss_fn = NTXEntLoss()
    loss = loss_fn(z_anchor, z_positive)
    print(loss)

