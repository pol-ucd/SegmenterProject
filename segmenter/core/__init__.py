from .random import randbool, randimage, randmask, randbool_like, randimage_like, randmask_like
from .seed import freeze_seed, unfreeze_seed
from .system import check_path_exists
from .config import Config, ConfigError
from .torch import get_default_device, get_default_device_type, set_default_device

__version__ = "0.0.1"
__author__ = "Pol Mac Aonghusa"
__email__ = "polmacaonghusa@gmail.com"

__all__ = ["randbool",
           "randbool_like",
           "randimage",
           "randimage_like",
           "randmask",
           "randmask_like",
           "freeze_seed",
           "unfreeze_seed",
           "check_path_exists",
           "Config",
           "ConfigError",
           "get_default_device_type",
           "get_default_device",
           "set_default_device"]

