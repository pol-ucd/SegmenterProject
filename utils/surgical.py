import random

import torch

from segmenter.masks import InstrumentMask, FluidMask, FoldMask, RandomShapeMask


class SurgicalMaskComposer:
    def __init__(self, shape, channels=3):
        self.H, self.W = shape
        self.C = channels
        self.instrument = InstrumentMask(shape=(self.H, self.W), channels=self.C)
        self.fluid = FluidMask(shape=(self.H, self.W), channels=self.C)
        self.fold = FoldMask(shape=(self.H, self.W), channels=self.C)
        self.shape = RandomShapeMask(shape=(self.H, self.W), channels=self.C)


    def generate_batch(self, batch_size):
        masks = []
        metadata = []

        for _ in range(batch_size):
            mask, info = self._generate_single()
            masks.append(mask)
            metadata.append(info)

        return torch.stack(masks), metadata

    def _generate_single(self):
        composite = torch.ones(self.H, self.W)
        info = {}

        # Randomize mask types
        use_instrument = random.random() < 0.5
        use_fluid = random.random() < 0.5
        use_fold = random.random() < 0.5
        use_shape = random.random() < 0.3

        if use_instrument:
            m = self.instrument()
            composite *= m
            info['instrument'] = (m == 0).nonzero(as_tuple=False)

        if use_fluid:
            m = self.fluid()
            composite *= m
            info['fluid'] = (m < 0.95).nonzero(as_tuple=False)

        if use_fold:
            m = self.fold()
            composite *= m
            info['fold'] = (m == 0).nonzero(as_tuple=False)

        if use_shape:
            m = self.shape()
            composite *= m
            info['shape'] = (m == 0).nonzero(as_tuple=False)

        # Multi-scale: downsample and upsample to blur edges
        composite = torch.nn.functional.avg_pool2d(composite.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
        composite = composite.clamp(0, 1)

        # Expand to channels
        final_mask = composite.unsqueeze(0).repeat(self.C, 1, 1)
        info['final_mask'] = (composite < 0.95).nonzero(as_tuple=False)

        return final_mask, info


