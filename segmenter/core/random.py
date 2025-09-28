from typing import Sequence, TypeVar

import torch
from torch import rand, randint

Probability = TypeVar('Probability', bound=float)

def is_valid_probability(p: Probability) -> bool:
    return 0 <= p <= 1

def randbool(size: Sequence[int], /, p: Probability = 0.5) -> torch.Tensor:
    """
    Samples a `torch.Tensor` with its shape as `size` filled with random booleans. The sampled tensor will have true values at possibility `p`.

    - Parameters:
        - size: A `Sequence` of the output tensor shape in `int`
        - p: A possibility in `float`
    - Returns: A sampled `torch.Tensor`
    """
    # check possibility value
    if not is_valid_probability(p):
        raise ValueError(f"Probability, 'p',  value must be in range [0, 1], got {p}.")

    # sampling
    return rand(size) < p

def randbool_like(t: torch.Tensor, /, p: float = 0.5) -> torch.Tensor:
    return randbool(t.shape, p=p)

def randimage(shape: Sequence[int], /) -> torch.Tensor:
    return randint(0, 255, shape).long()

def randimage_like(t: torch.Tensor) -> torch.Tensor:
    return randimage(t.shape)

def randmask(shape: Sequence[int], /, num_classes=2) -> torch.Tensor:
    return randint(0, num_classes, shape).long()

def randmask_like(t: torch.Tensor, /) -> torch.Tensor:
    return randmask_like(t.shape)