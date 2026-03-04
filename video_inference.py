"""
Video Inference Script using OptimizedSegmenter
- Processes video frames with temporal smoothing
- Supports keyframe-based processing for efficiency
- Outputs masks aligned with original video FPS
- Cross-platform: MPS, CUDA, CPU support
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from segmenter.core import Config
from segmenter.models import AugurSegformerSegmentation
from segmenter.models.optimized_inference import (
    OptimizedSegmenter, 
    InferenceConfig,
    FrameCache
)
from segmenter.utils import get_device, get_device_name, clear_cache
from segmenter.utils.mps_patch import patch_segformer_for_mps


def load_video(video_path: str) -> tuple:
    """Load video and return frames and metadata."""
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    return frames, (fps, width, height, total_frames)


def save_video(frames: list, output_path: str, metadata: tuple, fps: Optional[float] = None):
    """Save frames as video."""
    fps = fps or metadata[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (metadata[1], metadata[2]))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def process_video(
    video_path: str,
    model: AugurSegformerSegmentation,
    config: InferenceConfig,
    output_path: str,
    device: torch.device,
    save_masks: bool = True,
    overlay_alpha: float = 0.5,
    skip_frames: int = 1,
) -> dict:
    """Process video with temporal smoothing."""
    
    print(f"Loading video: {video_path}")
    frames, metadata = load_video(video_path)
    print(f"Video: {metadata[3]} frames, {metadata[0]:.2f} FPS, {metadata[1]}x{metadata[2]}")
    
    optimizer = OptimizedSegmenter(model, config)
    optimizer.to(device)
    
    frame_cache = FrameCache(max_size=30, similarity_threshold=0.95)
    
    output_masks = []
    output_overlays = []
    
    print(f"Processing {len(frames)} frames...")
    
    for i, frame in enumerate(tqdm(frames)):
        if i % skip_frames != 0:
            if output_masks:
                output_masks.append(output_masks[-1])
                if save_masks:
                    output_overlays.append(output_overlays[-1])
            continue
        
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        frame_tensor = frame_tensor.to(device)
        
        should_process = frame_cache.should_process(i, frame_tensor)
        
        if should_process:
            prob_map, binary_mask = optimizer.predict(
                frame_tensor / 255.0,
                apply_smoothing=True,
                apply_temporal=True
            )
            # B3: cache probability maps (continuous) not binary masks for accurate similarity detection
            frame_cache.add(i, prob_map.cpu())
        else:
            binary_mask = output_masks[-1] if output_masks else torch.zeros(1, 1, frame.shape[0], frame.shape[1])
        
        mask_np = binary_mask.squeeze().cpu().numpy()
        mask_np = (mask_np * 255).astype(np.uint8)
        
        output_masks.append(mask_np)
        
        if save_masks:
            mask_colored = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 1 - overlay_alpha, mask_colored, overlay_alpha, 0)
            output_overlays.append(overlay)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_masks:
        masks_path = output_dir / f"{Path(video_path).stem}_masks.mp4"
        save_video(output_masks, str(masks_path), metadata)
        print(f"Saved masks: {masks_path}")
        
        overlays_path = output_dir / f"{Path(video_path).stem}_overlay.mp4"
        save_video(output_overlays, str(overlays_path), metadata)
        print(f"Saved overlay: {overlays_path}")
    
    return {
        'num_frames': len(frames),
        'processed_frames': len(output_masks),
        'fps': metadata[0],
    }


def main():
    parser = argparse.ArgumentParser(description='Video Inference with Temporal Smoothing')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--no_temporal', action='store_true', help='Disable temporal smoothing')
    parser.add_argument('--temporal_alpha', type=float, default=0.7, help='Temporal smoothing strength')
    parser.add_argument('--blur_kernel', type=int, default=3, help='Blur kernel size (0 to disable)')
    parser.add_argument('--use_tta', action='store_true', help='Enable test-time augmentation')
    parser.add_argument('--skip_frames', type=int, default=1, help='Process every N frames')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='Overlay transparency')
    parser.add_argument('--save_masks', action='store_true', default=True, help='Save mask videos')
    
    args = parser.parse_args()
    
    patch_segformer_for_mps()
    
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading model...")
    base_model = AugurSegformerSegmentation(
        pretrained_model='nvidia/segformer-b2-finetuned-ade-512-512',
        num_classes=2,
        k=0,
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    
    inference_config = InferenceConfig(
        image_size=args.image_size,
        use_temporal_smoothing=not args.no_temporal,
        temporal_alpha=args.temporal_alpha,
        blur_kernel=args.blur_kernel,
        use_tta=args.use_tta,
    )
    
    results = process_video(
        video_path=args.video,
        model=base_model,
        config=inference_config,
        output_path=args.output,
        device=device,
        skip_frames=args.skip_frames,
        overlay_alpha=args.overlay_alpha,
        save_masks=args.save_masks,
    )
    
    print(f"\nProcessing complete!")
    print(f"  Frames: {results['processed_frames']}/{results['num_frames']}")
    print(f"  FPS: {results['fps']}")


if __name__ == '__main__':
    main()
