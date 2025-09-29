from typing import Tuple

import torchvision

from fluid import FluidMask
from instrument import InstrumentMask
from polygon import RandomShapeMask
from segmenter.masks.fold import FoldMask


def composite_occlusion(shape: Tuple,
                        channels: int,
                        num_shapes=2):
    """
    An example of combining multiple masks
    """
    im = InstrumentMask(shape=shape,channels=channels, num_shapes=num_shapes)
    fm = FluidMask(shape=shape,channels=channels, num_shapes=num_shapes)
    rm = RandomShapeMask(shape=shape,channels=channels, num_shapes=num_shapes)
    fom = FoldMask(shape=shape,channels=channels, num_shapes=num_shapes)

    mask1 = im()
    mask2 = fm()
    mask3 = rm()
    mask4 = fom()

    combined = (mask1 + mask2 + mask3 + mask4 > 0).float()
    return combined


if __name__ == '__main__':
    n_channels = 1
    b, c, h, w = 8, 3, 240, 320
    mask = composite_occlusion(shape=(b, c, h, w), channels=n_channels)

    assert mask.shape == (b, n_channels, h, w), "Something went wrong, check dimensions."

    trans = torchvision.transforms.ToPILImage()
    out = trans(mask[0])
    out.show()