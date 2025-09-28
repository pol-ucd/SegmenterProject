__author__ = 'Pol Mac Aonghusa'
__email__ = 'polmacaonghusa@gmail.com'
__version__ = '1.0'

from .base_mask import BaseMask
from .fluid import FluidMask
from .instrument import InstrumentMask
from .polygon import RandomShapeMask

__all__ = ['BaseMask',
           'RandomShapeMask',
           'InstrumentMask',
           'FluidMask']

