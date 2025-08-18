# SegmenterProject
Code and other material from the the segmentation project Hanija worked on.

# Data Transforms using Albumentations
The code supports multiclass segmentation with per‑pixel integer masks. So that it is 
in semantic segmentation territory: each mask pixel is a class index. That changes four things:

Model outputs shape becomes [B, C, H, W].
- B Batch size
- C Number of classes
- H image height
- W image width

The following setup is required.
- Masks are LongTensors of shape [B, H, W] (class indices).
- Albumentations must transform image and mask together (nearest‑neighbor for masks).
- Loss should be multiclass Jaccard/IoU/Dice and usually combined with Cross‑Entropy for stability.