"""
Model Quantization for Fast Inference
Post-training quantization and dynamic quantization for reduced model size and faster inference.
"""

import torch
import torch.nn as nn
from typing import Optional


class QuantizedSegmenter(nn.Module):
    """Wrapper for quantized models with preprocessing."""
    
    def __init__(
        self,
        model: nn.Module,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability map."""
        with torch.no_grad():
            logits = self.forward(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            return torch.softmax(logits, dim=1)[:, 1:2, :, :]


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to model for faster inference.
    
    Note: Dynamic quantization is applied to linear layers only.
    Works best with CPU inference.
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model


def apply_static_quantization(
    model: nn.Module,
    calibration_data: torch.Tensor,
) -> nn.Module:
    """Apply static quantization with calibration.
    
    Args:
        model: Model to quantize
        calibration_data: Sample input for calibration
    """
    model.eval()
    
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    torch.quantization.prepare(model, inplace=True)
    
    with torch.no_grad():
        for _ in range(10):
            model(calibration_data)
    
    torch.quantization.convert(model, inplace=True)
    
    return model


def create_int8_model(model: nn.Module, example_input: torch.Tensor):
    """Create INT8 quantized model with best available method."""
    
    model.eval()
    
    if torch.cuda.is_available():
        try:
            model.qconfig = torch.quantization.get_default_qconfig('tensorrt')
            torch.quantization.prepare(model, inplace=True)
            
            with torch.no_grad():
                model(example_input)
            
            quantized_model = torch.quantization.convert(model, inplace=False)
            print("Using TensorRT quantization")
            return quantized_model
        except:
            pass
    
    try:
        return apply_dynamic_quantization(model)
    except:
        print("Quantization failed, returning original model")
        return model


def benchmark_inference_speed(
    model: nn.Module,
    input_size: tuple = (1, 3, 512, 512),
    num_iterations: int = 100,
    warmup: int = 10,
    device: str = 'cpu',
) -> dict:
    """Benchmark model inference speed."""
    
    model.eval()
    model = model.to(device)
    
    dummy_input = torch.randn(input_size).to(device)
    
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    import time
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
    
    times = times[5:] if len(times) > 5 else times
    
    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'fps': 1.0 / (sum(times) / len(times)),
        'device': device,
        'input_size': input_size,
    }


def print_model_size(model: nn.Module) -> dict:
    """Print and return model size information."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    print(f"Model size: {size_mb:.2f} MB")
    print(f"  Parameters: {param_size / 1024 / 1024:.2f} MB")
    print(f"  Buffers: {buffer_size / 1024 / 1024:.2f} MB")
    
    return {
        'total_mb': size_mb,
        'parameters_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024,
    }
