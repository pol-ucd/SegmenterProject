import os

import torch

from nn.models import SegformerBinarySegmentation4

DATA_PATH="data/Arctur/Initial_frames_v2"
MODEL_PATH="best_segformer.pth"

if __name__ == '__main__':
    model_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    print(model_dict.keys())
    model = SegformerBinarySegmentation4(**model_dict)
    # model.eval()

