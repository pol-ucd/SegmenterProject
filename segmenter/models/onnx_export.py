"""
ONNX Export Utilities for Fast Inference
Optimized for real-time surgical video processing.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple


class ExportableSegmenter(nn.Module):
    """
    Wrapper that prepares model for ONNX export with:
    - Dynamic input size support
    - Fixed input shape for optimization
    - Pre/post-processing integration
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (512, 512),
        smooth_output: bool = True,
        smooth_kernel: int = 3,
    ):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.smooth_output = smooth_output
        
        if smooth_kernel > 0:
            self.smooth = nn.Conv2d(
                1, 1, 
                kernel_size=smooth_kernel, 
                padding=smooth_kernel // 2,
                bias=False
            )
            nn.init.constant_(self.smooth.weight, 1.0 / (smooth_kernel * smooth_kernel))
            for param in self.smooth.parameters():
                param.requires_grad = False
        else:
            self.smooth = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W] normalized to ImageNet stats
            
        Returns:
            Probability map [B, 1, H, W]
        """
        logits = self.model(x)
        
        if isinstance(logits, tuple):
            logits = logits[0]
        
        probs = torch.softmax(logits, dim=1)[:, 1:2, :, :]
        
        if self.smooth is not None and self.smooth_output:
            probs = self.smooth(probs)
        
        return probs


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (512, 512),
    opset_version: int = 17,
    simplify: bool = True,
    dynamic_axes: dict = None,
    force_cpu: bool = True,
):
    """
    Export segmentation model to ONNX format.
    
    Args:
        model: The segmentation model
        output_path: Path to save ONNX file
        input_size: Input resolution (H, W)
        opset_version: ONNX opset version
        simplify: Apply ONNX simplification
        dynamic_axes: Dict of dynamic axis specifications
        force_cpu: If True, move model to CPU for export (recommended for cross-platform compatibility)
    """
    model.eval()
    
    original_device = next(model.parameters()).device
    
    if force_cpu:
        model = model.cpu()
        print(f"Moving model from {original_device} to CPU for ONNX export")
    
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )
    
    print(f"Exported to ONNX: {output_path}")
    
    if simplify:
        try:
            import onnx
            from onnxsim import simplify
            
            onnx_model = onnx.load(output_path)
            model_simp, check = simplify(onnx_model)
            
            if check:
                onnx.save(model_simp, output_path)
                print(f"Simplified ONNX model saved to: {output_path}")
            else:
                print("ONNX simplification failed, keeping original")
        except ImportError:
            print("onnxsim not installed, skipping simplification")
        except Exception as e:
            print(f"ONNX simplification error: {e}")
    
    return output_path


def verify_onnx_model(onnx_path: str, torch_model: nn.Module, test_input: torch.Tensor):
    """Verify ONNX model produces similar output to PyTorch model."""
    import onnxruntime as ort
    
    ort_session = ort.InferenceSession(onnx_path)
    
    torch_model.eval()
    with torch.no_grad():
        torch_output = torch_model(test_input).numpy()
    
    ort_output = ort_session.run(None, {'input': test_input.numpy()})[0]
    
    max_diff = abs(torch_output - ort_output).max()
    mean_diff = abs(torch_output - ort_output).mean()
    
    print(f"ONNX vs PyTorch difference:")
    print(f"  Max: {max_diff:.6f}")
    print(f"  Mean: {mean_diff:.6f}")
    
    return max_diff < 1e-4


class ONNXInference:
    """Optimized ONNX Runtime inference wrapper."""
    
    def __init__(
        self,
        model_path: str,
        providers: list = None,
        intra_threads: int = 4,
        inter_threads: int = 0,
    ):
        if providers is None:
            providers = [
                ('CUDAExecutionProvider', {
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                ('CPUExecutionProvider', {
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'intra_op_num_threads': intra_threads,
                    'inter_op_num_threads': inter_threads,
                }),
            ]
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.input_shape = self.session.get_inputs()[0].shape
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference on single image."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image_np = image.numpy()
        
        output = self.session.run([self.output_name], {self.input_name: image_np})[0]
        
        return torch.from_numpy(output)
    
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference on batch of images."""
        images_np = images.numpy()
        
        output = self.session.run([self.output_name], {self.input_name: images_np})[0]
        
        return torch.from_numpy(output)


def create_optimized_inference_pipeline(
    model: nn.Module,
    checkpoint_path: str,
    output_dir: str = "exported_models",
    input_size: Tuple[int, int] = (512, 512),
):
    """Create complete optimized inference pipeline."""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    exportable = ExportableSegmenter(model, input_size=input_size)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / "segmenter_optimized.onnx"
    export_to_onnx(exportable, str(onnx_path), input_size=input_size)
    
    return str(onnx_path)
