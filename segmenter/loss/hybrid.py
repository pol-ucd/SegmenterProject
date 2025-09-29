from typing import Dict

import torch
from torch import nn as nn

from segmenter.loss import BaseLoss, LossFactory


class HybridLoss(BaseLoss):
    """
    Combines multiple loss functions with configurable weights.
    The list of losses to use and their weights are passed in a dictionary.
    """

    def __init__(self, loss_configs: Dict[str, Dict[str, float]]):
        """
        Args:
            loss_configs (Dict[str, Dict[str, float]]):
                A dictionary where keys are loss names and values are dictionaries
                containing 'weight' and any other loss-specific arguments.
                Example: {'dice': {'weight': 0.5}, 'ce': {'weight': 0.5}}.
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict()
        self.weights = {}

        for name, config in loss_configs.items():
            weight = config.pop('weight', 1.0)
            self.weights[name] = weight
            self.loss_functions[name] = LossFactory.create(name, **config)
            # logging.info(f"Initialized loss '{name}' with weight {weight} and config {config}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _ = super().setup(pred, target)
        device = pred.device

        total_loss = torch.tensor(0.0, device=device)
        total_weight = sum(v for v in self.weights.values())
        for name, loss_fn in self.loss_functions.items():
            current_loss = loss_fn(pred,
                                   self.targets)
            total_loss += self.weights[name] * current_loss

        return total_loss/total_weight
