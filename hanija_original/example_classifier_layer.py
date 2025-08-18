import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from transformers import SegformerConfig, SegformerForSemanticSegmentation

"""
Code taken from finalcode.py and adapted to insert the new Sequentia()
layer as the classifier
"""
config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
config.num_labels = 1
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512", config=config, ignore_mismatched_sizes=True)


classifier_in_size = model.decode_head.linear_fuse.out_channels
model.decode_head.classifier = nn.Sequential(
    nn.Conv2d(classifier_in_size, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 1, kernel_size=1)
)

"""
Dummy test data to check everything runs correctly
"""
test_input = torch.randn(1, 3, 512, 512)
test_mask = torch.randint(0, 1, (1, 1, 512, 512))   # Mask should be shape: (1, 1, 512, 512)

logits = model(test_input).logits

"""
Segformer downscales, so rescale to original 
"""
test_pred = F.interpolate(logits,
                          size=test_input.shape[2:],
                          mode='bilinear', align_corners=False)

"""
Check dimensions are correct for a mask of the input
"""
print("test_pred shape: ", test_pred.shape)
# Verify output properties
if test_pred.shape == test_mask.shape:
    print("SUCCESS: Output shape is as expected for binary semantic segmentation.")
else:
    print("FAIL: Output shape is NOT as expected. Please check the implementation.")
print("\n2. Example complete.")

"""
Print a summary of the model structure
"""
summary(model)