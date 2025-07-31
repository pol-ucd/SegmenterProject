from torch import nn as nn
from torch.nn import functional as F
from transformers import SegformerModel
from torchinfo import summary   # Required for testing only


class SegformerBinarySegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        hidden_dim = self.backbone.config.hidden_sizes[-1]
        self.decode_head = nn.Sequential(nn.Conv2d(hidden_dim,
                                                   out_channels=256,
                                                   kernel_size=3,
                                                   padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(in_channels=256,
                                                   out_channels=1,
                                                   kernel_size=1)
                                         )

    def forward(self, pixel_values):
        features = self.backbone(pixel_values=pixel_values).last_hidden_state  # [B, C, H/32, W/32]
        logits = self.decode_head(features)  # [B, 1, H/32, W/32]
        logits = F.interpolate(logits,
                               size=pixel_values.shape[2:],
                               mode='bilinear',
                               align_corners=False)
        return logits  # [B, 1, 512, 512]


if __name__ == '__main__':
    model = SegformerBinarySegmentation()
    summary(model)