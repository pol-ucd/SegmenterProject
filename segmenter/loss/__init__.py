__version__ = "0.0.1"
__author__ = "Pol Mac Aonghusa"
__email__ = "polmacaonghusa@gmail.com"

from .losses import (
    DiceLoss,
    FocalLoss,
    BoundaryLoss,
    IoULoss,
    TverskyLoss,
    ComboLoss,
    TemporalConsistencyLoss,
    MultiScaleLoss,
    get_loss_function,
)


__all__ = [
    "DiceLoss",
    "FocalLoss",
    "BoundaryLoss",
    "IoULoss",
    "TverskyLoss",
    "ComboLoss",
    "TemporalConsistencyLoss",
    "MultiScaleLoss",
    "get_loss_function",
]