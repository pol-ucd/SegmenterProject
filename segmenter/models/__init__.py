__author__ = "Pol Mac Aonghusa"
__email__ = "polmacaonghusa@gmail.com"

from .base import MedianPool2d, SegformerModelError, SegformerBackbone, SupervisedSegFormer, get_pretrained_model
from .models import BaseSegmenter, AugurSegformerSegmentation, CustomSegformerDecodeHead
from .optimized_inference import OptimizedSegmenter, InferenceConfig, TemporalSmoother, FrameCache, FastGaussianBlur

__all__ = [
    'MedianPool2d', 'SegformerModelError', 'SegformerBackbone',
    'SupervisedSegFormer',
    'get_pretrained_model', 'BaseSegmenter', 'AugurSegformerSegmentation', 
    'CustomSegformerDecodeHead',
    'OptimizedSegmenter', 'InferenceConfig', 'TemporalSmoother', 'FrameCache', 'FastGaussianBlur',
]

try:
    from .onnx_export import ExportableSegmenter, export_to_onnx, ONNXInference, create_optimized_inference_pipeline
    from .quantization import QuantizedSegmenter, apply_dynamic_quantization, benchmark_inference_speed
    
    __all__.extend([
        'ExportableSegmenter', 'export_to_onnx', 'ONNXInference', 'create_optimized_inference_pipeline',
        'QuantizedSegmenter', 'apply_dynamic_quantization', 'benchmark_inference_speed',
    ])
except ImportError:
    pass