import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss with a tunable temperature parameter (tau).

    Args:
        margin (float): Distance margin 'm'. Determines how far apart
                        dissimilar samples should be pushed.
        tau (float): Temperature parameter. Used to scale the distance metric.
                     A larger tau makes the loss more sensitive to small changes
                     in distance, similar to how it works in triplet/NT-Xent loss.
    """

    def __init__(self, margin=1.0, tau=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.tau = tau

    def forward(self, embedding_1, embedding_2, label):
        """
        Calculates the contrastive loss for a batch of embeddings.

        Args:
            embedding_1 (torch.Tensor): Tensor of the first set of embeddings (e.g., N x D).
            embedding_2 (torch.Tensor): Tensor of the second set of embeddings (e.g., N x D).
            label (torch.Tensor): Tensor of labels (0 for dissimilar, 1 for similar), size N.

        Returns:
            torch.Tensor: Scalar loss value (mean loss over the batch).
        """
        # Calculate the Euclidean distance (L2 norm) between the embeddings
        # The result is a tensor of shape (N,)
        euclidean_distance = F.pairwise_distance(embedding_1, embedding_2, p=2)

        # Apply the temperature parameter to scale the distance
        # A simple way to incorporate tau is by scaling the distance D -> D / tau
        # or D^2 -> D^2 / tau, making the loss function shallower or steeper.
        scaled_distance_sq = (euclidean_distance ** 2) / self.tau

        # Loss for similar pairs (y=1):
        # L_s = 1/2 * D^2 / tau
        loss_similar = 0.5 * label.float() * scaled_distance_sq

        # Loss for dissimilar pairs (y=0):
        # L_d = 1/2 * max(0, m - D)^2
        # Note: We must work with the *distance* (D) for the margin, then square it.

        # Margin term: max(0, m - D)
        distance_for_dissimilar = euclidean_distance

        # We only care about dissimilar pairs (1 - label)
        dissimilar_term = (1 - label.float())

        # Calculate the squared distance when distance < margin
        max_of_zero_and_margin_minus_distance = torch.clamp(self.margin - distance_for_dissimilar, min=0.0)

        # Loss for dissimilar pairs: 1/2 * (1-y) * max(0, m - D)^2
        loss_dissimilar = 0.5 * dissimilar_term * (max_of_zero_and_margin_minus_distance ** 2)

        # Total loss is the mean of the losses for all pairs
        contrastive_loss = torch.mean(loss_similar + loss_dissimilar)

        return contrastive_loss


# ----------------------------------------------------------------------
## Usage Example
# ----------------------------------------------------------------------

# Dummy parameters
BATCH_SIZE = 16
EMBEDDING_DIM = 64

# Create random embeddings and labels for a batch
# h1, h2: (Batch_Size, Embedding_Dim)
h1 = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
h2 = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
# label: (Batch_Size,) - 1 for similar, 0 for dissimilar
# We'll make half similar (1) and half dissimilar (0)
labels = torch.cat([torch.ones(BATCH_SIZE // 2), torch.zeros(BATCH_SIZE // 2)]).long()

# Initialize the loss function
# Set an initial temperature (tau) of 1.0 (no effect on the distance magnitude)
# You can tune tau (e.g., 0.5, 2.0, 10.0) to see how it affects training.
criterion = ContrastiveLoss(margin=1.0, tau=1.0)

# Calculate the loss
loss = criterion(h1, h2, labels)

print(f"Calculated Contrastive Loss: {loss.item():.4f}")