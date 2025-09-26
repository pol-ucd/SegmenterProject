from torch import nn as nn

from segmenter.loss import DiceLoss, IoULoss, FocalLoss, TverskyLoss
from segmenter.loss.boundary_sdf import BoundarySDFLoss
from segmenter.loss.soft_chamfer import SoftChamferLoss


class LossFactory:
    """
    A factory class to create instances of loss functions by name.
    This centralizes loss function creation and configuration.
    """
    _loss_map = {
        'dice': DiceLoss,
        'iou': IoULoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'boundary_sdf': BoundarySDFLoss,
        'soft_chamfer': SoftChamferLoss,
        'ce': nn.CrossEntropyLoss,
        'bce': nn.BCEWithLogitsLoss
    }

    @staticmethod
    def create(loss_name: str, **kwargs) -> nn.Module:
        """
        Creates a loss function instance.
        Args:
            loss_name (str): The name of the loss function.
            **kwargs: Additional arguments for the loss function's constructor.
        Returns:
            nn.Module: An instance of the requested loss function.
        Raises:
            ValueError: If the loss name is not supported.
        """
        if loss_name not in LossFactory._loss_map:
            raise ValueError(f"Loss '{loss_name}' not supported. "
                             f"Supported losses are: {list(LossFactory._loss_map.keys())}")
        return LossFactory._loss_map[loss_name](**kwargs)
