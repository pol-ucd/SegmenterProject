"""
CoreML Export for Apple Devices (iOS, macOS, Vision Pro)
Optimized for MPS/Metal GPU inference on Apple hardware.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class CoreMLSegmenter(nn.Module):
    """Wrapper for CoreML export with preprocessing."""
    
    def __init__(
        self,
        model: nn.Module,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Probability map [B, 1, H, W]
        """
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        
        x = (x - mean) / std
        
        logits = self.model(x)
        
        if isinstance(logits, tuple):
            logits = logits[0]
        
        probs = torch.softmax(logits, dim=1)[:, 1:2, :, :]
        
        return probs


def export_to_coreml(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (512, 512),
    input_name: str = "input_image",
    output_name: str = "output_mask",
    minimum_deployment_target: str = "15",
) -> str:
    """
    Export model to CoreML format for Apple devices.
    
    Args:
        model: Segmentation model
        output_path: Path to save .mlmodel file
        input_size: Input resolution (H, W)
        input_name: Name for input tensor
        output_name: Name for output tensor
        minimum_deployment_target: Minimum iOS/macOS version
        
    Returns:
        Path to saved CoreML model
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools not installed. Install with: pip install coremltools")
    
    model.eval()
    
    example_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    traced_model = torch.jit.trace(model, example_input)
    
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name=input_name,
                shape=(1, 3, input_size[0], input_size[1]),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.ImageType(
                name=output_name,
                color_layout=ct.colorlayout.GRAYSCALE,
            )
        ],
        minimum_deployment_target=minimum_deployment_target,
    )
    
    coreml_model.save(output_path)
    print(f"Exported to CoreML: {output_path}")
    
    return output_path


def create_universal_apple_model(
    model: nn.Module,
    output_dir: str = "exported_models",
    input_size: Tuple[int, int] = (512, 512),
):
    """Create both CoreML and ONNX models for Apple devices."""
    import os
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    example_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    traced = torch.jit.trace(model, example_input)
    
    coreml_path = output_dir / "segmenter.mlpackage"
    
    try:
        import coremltools as ct
        
        coreml_model = ct.convert(
            traced,
            inputs=[
                ct.ImageType(
                    name="input_image",
                    shape=(1, 3, input_size[0], input_size[1]),
                    scale=1.0 / 255.0,
                )
            ],
            outputs=[ct.FeatureType("output_mask")],
        )
        
        coreml_model.save(str(coreml_path))
        print(f"CoreML: {coreml_path}")
    except ImportError:
        print("coremltools not installed, skipping CoreML export")
    
    onnx_path = output_dir / "segmenter.onnx"
    torch.onnx.export(
        traced,
        example_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    )
    print(f"ONNX: {onnx_path}")
    
    return {
        'coreml': str(coreml_path),
        'onnx': str(onnx_path),
    }
