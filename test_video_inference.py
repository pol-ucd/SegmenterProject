"""
Quick video inference test script.
Tests a small sample of frames from a video to verify the inference pipeline.
"""

import argparse
import cv2
import torch
import time
from pathlib import Path

from segmenter.models import AugurSegformerSegmentation, OptimizedSegmenter, InferenceConfig
from segmenter.utils import get_device, get_device_name, clear_cache


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_frames_sequential(video_path: str, num_frames: int = 10):
    """Load frames sequentially from video (more reliable than random access)."""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    step = max(1, total_frames // num_frames)
    
    frames = []
    frame_idx = 0
    
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
        
        if frame_idx >= total_frames:
            break
    
    cap.release()
    
    return frames, (total_frames, fps, width, height)


def preprocess_frame(frame, target_size=512):
    """Preprocess frame for model input. Uses direct resize to match training."""
    # B2: direct resize (no letterbox) to match training augmentation pipeline
    resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0
    # Return [0,1] range — OptimizedSegmenter.preprocess() handles ImageNet normalisation
    return tensor.permute(2, 0, 1).unsqueeze(0)


def compute_inter_frame_iou(prev_mask: torch.Tensor, curr_mask: torch.Tensor) -> float:
    """Compute IoU between consecutive frames to measure temporal consistency.
    
    High IoU (>0.95) indicates stable predictions.
    Low IoU indicates flickering or unstable predictions.
    """
    prev_binary = (prev_mask > 0.5).float()
    curr_binary = (curr_mask > 0.5).float()
    
    intersection = (prev_binary * curr_binary).sum()
    union = prev_binary.sum() + curr_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def test_video_inference(
    video_path: str,
    checkpoint_path: str = None,
    num_frames: int = 10,
    device: str = None,
    use_temporal: bool = True,
    use_smoothing: bool = True,
):
    """Test video inference pipeline."""
    
    if device is None:
        device = get_device()
    
    print(f"=== Video Inference Test ===")
    print(f"Video: {video_path}")
    print(f"Device: {device} ({get_device_name(device)})")
    print(f"Testing {num_frames} frames")
    print()
    
    print("Loading model...")
    model = AugurSegformerSegmentation(
        pretrained_model='nvidia/segformer-b2-finetuned-ade-512-512',
        num_classes=2,
    )
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("Using pretrained model (no checkpoint)")
    
    model = model.to(device)
    model.eval()
    
    config = InferenceConfig(
        use_temporal_smoothing=use_temporal,
        temporal_alpha=0.7,
        blur_kernel=3 if use_smoothing else 0,
        use_amp=True,
    )
    
    optimizer = OptimizedSegmenter(model, config)
    
    print("Loading video frames...")
    frames, metadata = load_frames_sequential(video_path, num_frames)
    total_frames, fps, width, height = metadata
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    print(f"Loaded {len(frames)} frames")
    print()
    
    print("Running inference...")
    clear_cache(device)
    
    times = []
    inter_frame_ious = []
    prev_mask = None
    
    for i, frame in enumerate(frames):
        frame_tensor = preprocess_frame(frame).to(device)
        
        start = time.perf_counter()
        with torch.no_grad():
            prob, mask = optimizer.predict(frame_tensor, apply_smoothing=True, apply_temporal=True)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        
        if prev_mask is not None:
            inter_iou = compute_inter_frame_iou(prev_mask, mask)
            inter_frame_ious.append(inter_iou)
        
        prev_mask = mask.clone()
        
        mask_np = mask.squeeze().cpu().numpy()
        foreground_pixels = (mask_np > 0.5).sum()
        total_pixels = mask_np.size
        coverage = foreground_pixels / total_pixels * 100
        
        iou_str = f", inter-frame IoU={inter_frame_ious[-1]:.3f}" if inter_frame_ious else ""
        print(f"  Frame {i+1}/{len(frames)}: {elapsed*1000:.1f}ms, coverage={coverage:.1f}%{iou_str}")
    
    avg_time = sum(times) / len(times) if times else 0
    fps_est = 1.0 / avg_time if avg_time > 0 else 0
    avg_inter_iou = sum(inter_frame_ious) / len(inter_frame_ious) if inter_frame_ious else 0.0
    
    print()
    print(f"=== Results ===")
    print(f"Average inference time: {avg_time*1000:.1f}ms")
    print(f"Estimated FPS: {fps_est:.1f}")
    print(f"Min time: {min(times)*1000:.1f}ms" if times else "N/A")
    print(f"Max time: {max(times)*1000:.1f}ms" if times else "N/A")
    print(f"Inter-frame IoU: {avg_inter_iou:.4f}" if inter_frame_ious else "N/A")
    
    return {
        'avg_time_ms': avg_time * 1000,
        'fps': fps_est,
        'inter_frame_iou': avg_inter_iou,
        'num_frames': len(frames),
    }


def main():
    parser = argparse.ArgumentParser(description='Quick video inference test')
    parser.add_argument('--video', type=str, default='videos/0000.mp4', help='Video path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to test')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/mps/cpu)')
    parser.add_argument('--no_temporal', action='store_true', help='Disable temporal smoothing')
    parser.add_argument('--no_smoothing', action='store_true', help='Disable Gaussian smoothing')
    
    args = parser.parse_args()
    
    test_video_inference(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        num_frames=args.num_frames,
        device=args.device,
        use_temporal=not args.no_temporal,
        use_smoothing=not args.no_smoothing,
    )


if __name__ == '__main__':
    main()
