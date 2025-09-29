import torch
import torch.nn.functional as F
from torch import nn


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

