import torch
from torch import nn as nn
from torch.nn import functional as F


def msn_loss(online_preds, target_protos, temperature=0.1):
    """
    Calculates the Mixed Siamese Network (MSN) loss.
    - online_preds: Predictions from the online network for MASKED patches.
    - target_protos: Representations from the target network for ALL patches.
    - temperature: Controls the sharpness of the target distribution.
    """
    # Normalize the prototypes and predictions
    online_preds = F.normalize(online_preds, dim=1)
    target_protos = F.normalize(target_protos, dim=1)

    # Calculate similarity scores between each masked patch and all target patches
    # Shape: (num_masked_patches, num_target_patches)
    similarity_matrix = torch.matmul(online_preds, target_protos.t())

    # Sharpen the target distribution and compute the loss
    # The target is the softmax over similarities with the target prototypes
    targets = F.softmax(similarity_matrix / temperature, dim=1)

    # The prediction is the log-softmax over the same similarities
    predictions = F.log_softmax(similarity_matrix, dim=1)

    # Cross-entropy loss
    loss = - (targets * predictions).sum(dim=1)
    return loss.mean()


class MSNLoss(nn.Module):
    def __init__(self, margin=1.0, tau=0.1):
        self.tau = tau
        self.margin = margin

    def __call__(self, online_preds, target_protos, temperature=0.1):
        return msn_loss(online_preds, target_protos, temperature)

    def forward(self, online_preds, target_protos, temperature=0.1):
        return msn_loss(online_preds, target_protos, temperature)


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

#
# class InfoNCELoss(nn.Module):
#     """
#     SimCLR/MoCo-style InfoNCE Loss for contrastive learning.
#     This simplifies the loss to focus on bringing the positive pair (Anchor vs Occluded) closer.
#     """
#
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
#
#     def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
#         """
#         Calculates the InfoNCE loss for a single positive pair (anchor, positive).
#         The negative samples are implicitly all other samples in the batch.
#         :param z_anchor: Anchor embeddings [B, D].
#         :param z_positive: Positive embeddings [B, D].
#         :return: Scalar loss tensor.
#         """
#         # Normalize embeddings
#         z_anchor = F.normalize(z_anchor, dim=1)
#         z_positive = F.normalize(z_positive, dim=1)
#
#         # Concatenate anchor and positive embeddings to form the main batch
#         # [2B, D]
#         features = torch.cat([z_anchor, z_positive], dim=0)
#
#         # Compute cosine similarity matrix: [2B, 2B]
#         similarity_matrix = torch.matmul(features, features.T) / self.temperature
#
#         # Create mask for positive pairs: 1 for positive, 0 otherwise
#         batch_size = z_anchor.shape[0]
#         mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
#
#         # The positive pairs are at (i, i+B) and (i+B, i)
#         # 1. Anchor i to Positive i+B
#         pos_mask_1 = torch.roll(torch.eye(batch_size, device=features.device), shifts=batch_size, dims=1)
#         # 2. Positive i+B to Anchor i
#         pos_mask_2 = torch.roll(torch.eye(batch_size, device=features.device), shifts=batch_size, dims=0)
#
#         # Combine into the full mask for the 2B x 2B matrix
#         # [B, B] [B, B]
#         # [B, B] [B, B]
#         # We need the positive pair connections:
#         # A_i -> P_i and P_i -> A_i
#
#         positive_pairs_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
#
#         # Anchor to Positive (i, i+B)
#         positive_pairs_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=features.device)
#         # Positive to Anchor (i+B, i)
#         positive_pairs_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=features.device)
#
#         # Exclude self-similarities (diagonal)
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#         positive_pairs_mask = positive_pairs_mask[~mask].view(positive_pairs_mask.shape[0], -1)
#
#         # Select the similarities for the positive pairs
#         positives = similarity_matrix[positive_pairs_mask].view(2 * batch_size, -1)
#
#         # The numerator of the InfoNCE loss (similarity with positive)
#         logits = positives
#
#         # All other samples (excluding self) are negatives
#         labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
#
#         # InfoNCE Loss: -log( exp(pos) / sum(exp(all)) )
#         # Using CrossEntropyLoss with logits and labels of 0 is equivalent to this:
#         loss = F.cross_entropy(logits, labels)
#
#         return loss


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
        :param z_anchor: Anchor embeddings [B, D].
        :param z_positive: Positive embeddings [B, D].
        :return: Scalar InfoNCE loss.
        """
        B = z_anchor.shape[0]

        # 1. Normalize embeddings (Crucial for cosine similarity)
        z_anchor = F.normalize(z_anchor, dim=1)
        z_positive = F.normalize(z_positive, dim=1)

        # Concatenate both views to form the full batch of features: [2B, D]
        features = torch.cat([z_anchor, z_positive], dim=0)

        # 2. Compute full cosine similarity matrix: [2B, 2B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 3. Create a mask to remove self-similarities (diagonal)
        # This mask will be used to filter the logits matrix.
        mask_diag = torch.eye(2 * B, dtype=torch.bool, device=features.device)

        # 4. Separate positive and negative logits

        # Positives (Numerator): Sim(A_i, P_i) and Sim(P_i, A_i)
        # These are at (i, i+B) and (i+B, i) in the similarity matrix.
        positives = torch.cat([
            similarity_matrix[:B, B:].diag(),  # A_i -> P_i
            similarity_matrix[B:, :B].diag()  # P_i -> A_i
        ], dim=0).view(2 * B, 1)

        # Negatives (Denominator): All other similarities, excluding self-similarity.
        # This matrix still contains the positive similarity, but we will put it in front.
        # First, remove the diagonal (self-similarity)
        # negatives_and_positives = similarity_matrix[~mask_diag].view(2 * B, -1)

        # Now, `negatives_and_positives` is a [2B, 2B-1] matrix.
        # For each anchor, one column is the positive similarity and the rest are negatives.

        # 5. Construct the final logits matrix: [Positives | Negatives]
        # We concatenate the separated `positives` with the filtered matrix,
        # then remove the duplicate positive similarity from the second part.

        # The indices of the positive pair in the flattened matrix (2B, 2B-1) are complex.
        # We must create a mask to remove the positive pair from the `negatives_and_positives` matrix.

        # A robust way is to use the full positive mask to select what to KEEP.

        # Mask for the positive pair connections in the (2B, 2B) matrix:
        mask_pos = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        mask_pos[:B, B:] = torch.eye(B, dtype=torch.bool, device=features.device)
        mask_pos[B:, :B] = torch.eye(B, dtype=torch.bool, device=features.device)

        # Combine masks: remove both the diagonal AND the positive connections
        mask_neg = ~(mask_diag | mask_pos)

        # Select the true negatives from the full similarity matrix: [2B, 2B-2]
        negatives = similarity_matrix[mask_neg].view(2 * B, -1)

        # Final logits matrix: [Positive Logit | All Negative Logits] -> [2B, 2B-1]
        logits = torch.cat([positives, negatives], dim=1)

        # The label for the positive pair is always 0 (it is in the 0th column)
        labels = torch.zeros(2 * B, dtype=torch.long, device=features.device)

        # 6. Apply Cross Entropy Loss (equivalent to InfoNCE)
        loss = F.cross_entropy(logits, labels)

        return loss