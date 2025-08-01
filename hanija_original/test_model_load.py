import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

device = 'cpu'
pt_file = '/Users/polmacaonghusa/Downloads/b_model (1).pt'

if __name__ == '__main__':
    config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
    config.num_labels = 1
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512",
                                                             config=config,
                                                             ignore_mismatched_sizes=True
                                                             )
    model.classifier = nn.Sequential(
        nn.Conv2d(config.hidden_sizes[-1], 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 1, kernel_size=1)
    )

    model.load_state_dict(torch.load(pt_file, map_location=device))

    model.load_state_dict(torch.load(pt_file,
                                     map_location=device))
