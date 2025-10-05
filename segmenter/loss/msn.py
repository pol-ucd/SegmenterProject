import torch
from torch import nn as nn
from torch.nn import functional as F


class MSNLoss(nn.Module):
    """
    Implements the Mean-Shifted Network (MSN) objective, minimizing
    KL divergence between online predictions (Q) and centered target assignments (P).
    """

    def __init__(self, temperature: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        # 'temperature' controls the sharpness of the target distribution (P).
        self.temperature = temperature

        # Center Momentum (m): Controls the speed of the EMA update.
        # MSN often uses m close to 0 (e.g., 0.1), while MoCo/DINO use m close to 1.
        self.center_momentum = center_momentum

        # Initialize target_center as a persistent buffer, not a trainable parameter.
        self.register_buffer("target_center", None)

    @torch.no_grad()
    def update_center(self, target_protos_current_batch: torch.Tensor):
        """
        Updates the target center using an Exponential Moving Average (EMA).
        This method must be called once per iteration on the target network's outputs.
        """
        # Ensure target_protos are detached (they should be, but safety check) and normalized
        target_protos = F.normalize(target_protos_current_batch.detach(), dim=1)

        # Calculate the mean of the current batch's target prototypes
        current_batch_center = target_protos.mean(dim=0, keepdim=True)

        # Initialize the center if it's the first run
        if self.target_center is None:
            self.target_center = current_batch_center.clone()
            return

        # EMA Update: c_new = (m) * c_old + (1 - m) * c_batch_mean
        new_center = self.target_center.clone() * self.center_momentum + \
                     current_batch_center * (1.0 - self.center_momentum)

        # Copy the updated value back to the registered buffer
        self.target_center.copy_(new_center)

    def forward(self, online_preds: torch.Tensor, target_protos: torch.Tensor) -> torch.Tensor:
        """
        Calculates the KL divergence between the online network's predictions (Q)
        and the target network's assignments (P).

        :param online_preds: Predictions from the online network for MASKED patches [N_masked, D].
        :param target_protos: Representations from the target network for ALL patches [N_all, D].
        :return: Scalar loss tensor (mean KL divergence).
        """

        # 1. Normalization
        online_preds = F.normalize(online_preds, dim=1)
        target_protos = F.normalize(target_protos, dim=1)

        # 2. Mean-Shift / Centering (The critical MSN step)
        # If center is not initialized, update_center should be called first,
        # but we use a robust check here.
        if self.target_center is None:
            # If the center hasn't been initialized, use the current batch mean as a proxy
            centered_target_protos = target_protos - target_protos.mean(dim=0, keepdim=True)
        else:
            # Subtract the stabilized moving average center from all target prototypes
            centered_target_protos = target_protos - self.target_center

        # Similarity Matrix: Sim(Online_Masked, Centered_Target_All)
        # Shape: (N_masked, N_all)

        similarity_matrix = torch.matmul(online_preds,
                                         centered_target_protos.transpose(-1, -2))

        # 4. Target Distribution (P) - Sharpened Softmax
        # P = softmax(Sim / temperature). This is the 'teacher' signal.
        targets = F.softmax(similarity_matrix / self.temperature, dim=1)

        # 5. Prediction Distribution (log Q) - Log Softmax
        # log Q = log(softmax(Sim)). This is the 'student' prediction.
        predictions = F.log_softmax(similarity_matrix, dim=1)

        # 6. KL Divergence / Cross-Entropy
        # L = - sum(P * log Q) -> Minimizes KL(P || Q).
        loss = - (targets * predictions).sum(dim=1)

        return loss.mean()


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
        # 1. Normalize embeddings before calculating loss (as per SimSiam implementation)
        p1 = F.normalize(p1, dim=1)
        z2_detached = F.normalize(z2_detached, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1_detached = F.normalize(z1_detached, dim=1)

        # 2. Calculate the two symmetric loss terms (MSE)
        # Term 1: Prediction from view 1 vs Target from detached view 2
        loss1 = self.mse_loss(p1, z2_detached)

        # Term 2: Prediction from view 2 vs Target from detached view 1
        loss2 = self.mse_loss(p2, z1_detached)

        # 3. Total Symmetric Loss (Averaged)
        total_loss = 0.5 * (loss1 + loss2)

        return total_loss


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

        logits = similarity_matrix.reshape(B*B, -1)

        # The label for the positive pair is always 0 (it is in the 0th column)
        labels = torch.eye(B, dtype=torch.long, device=logits.device).reshape(B*B)

        # Apply Cross Entropy Loss (equivalent to InfoNCE)
        loss = F.cross_entropy(logits, labels)

        return loss


if __name__ == '__main__':
    temperature = 0.2
    B, C, H, W = 4, 11, 512, 512

    z_anchor = torch.randint(-255, 255, (B, C, H, W)).float() / 255
    z_positive = torch.randint(-255, 255, (B, C, H, W)).float() / 255


    loss = InfoNCELoss(temperature=temperature)(z_anchor, z_positive)
    print(loss)
