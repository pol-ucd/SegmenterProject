__version__ = "0.0.1"
__author__ = "Pol Mac Aonghusa"
__email__ = "polmacaonghusa@gmail.com"

EPSILON = 1e-06

from .base import BaseLoss, LossException
from .utils import sobel_grad, one_hot
from .distance import DistanceTransform2D

from .soft_chamfer import SoftChamferLoss
from .boundary_sdf import BoundarySDFLoss
from .dice import DiceLoss
from .focal import FocalLoss
from .iou import IoULoss
from .msn import MSNLoss, NTXentLoss
from .tversky import TverskyLoss
from .maskedcosine import (MaskedCosineSimilarityLoss, masked_cosine_similarity_loss,
                           EncodingCosineSimilarityLoss, enc_cosine_similarity_loss)

from .factory import LossFactory
from .hybrid import HybridLoss


__all__ = [
    "BaseLoss",
    "LossException",
    "sobel_grad",
    "one_hot",
    "EPSILON",
    "BoundarySDFLoss",
    "SoftChamferLoss",
    "DistanceTransform2D",
    "FocalLoss",
    "TverskyLoss",
    "IoULoss",
    "DiceLoss",
    "IoULoss",
    "MSNLoss",
    "NTXentLoss",
    "LossFactory",
    "HybridLoss",
    "MaskedCosineSimilarityLoss",
    "masked_cosine_similarity_loss",
    "EncodingCosineSimilarityLoss",
    "enc_cosine_similarity_loss"
]