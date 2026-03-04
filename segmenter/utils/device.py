"""
Cross-platform device utilities for MPS/CUDA/CPU support.
Provides safe device handling and detection across platforms.
"""

import torch
from typing import Optional, Tuple
from enum import Enum


class DeviceType(Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def get_device() -> torch.device:
    """Auto-detect the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_device_type(device: Optional[torch.device] = None) -> DeviceType:
    """Get the device type enum from a torch device."""
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        return DeviceType.CUDA
    elif device.type == 'mps':
        return DeviceType.MPS
    return DeviceType.CPU


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon GPU) is available."""
    return torch.backends.mps.is_available()


def is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA GPU) is available."""
    return torch.cuda.is_available()


def get_device_name(device: Optional[torch.device] = None) -> str:
    """Get a human-readable device name."""
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    elif device.type == 'mps':
        return "Apple Silicon (MPS)"
    return "CPU"


def get_autocast_device_type(device: Optional[torch.device] = None) -> str:
    """Get the device type string for torch.amp.autocast.
    
    Args:
        device: The device to use. If None, auto-detects.
        
    Returns:
        'cuda', 'mps', or 'cpu' for autocast
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        return 'cuda'
    elif device.type == 'mps':
        return 'mps'
    return 'cpu'


def supports_amp(device: Optional[torch.device] = None) -> bool:
    """Check if the device supports automatic mixed precision (AMP)."""
    if device is None:
        device = get_device()
    return device.type in ('cuda', 'mps')


def move_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Safely move a model to the specified device.
    
    Handles different device types safely.
    """
    return model.to(device)


def sync_device(device: Optional[torch.device] = None):
    """Synchronize the device (no-op for CPU/MPS, sync for CUDA)."""
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize()


class DeviceContext:
    """Context manager for temporary device operations."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.original_device = None
    
    def __enter__(self):
        self.original_device = torch.cuda.current_device() if self.device.type == 'cuda' else None
        return self
    
    def __exit__(self, *args):
        if self.original_device is not None:
            torch.cuda.set_device(self.original_device)


def get_optimal_batch_size(base_batch_size: int, device: Optional[torch.device] = None) -> int:
    """Adjust batch size based on device memory constraints.
    
    Args:
        base_batch_size: The baseline batch size
        device: Target device
        
    Returns:
        Adjusted batch size for the device
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        return base_batch_size
    elif device.type == 'mps':
        return max(1, base_batch_size // 2)
    return max(1, base_batch_size // 4)


def prepare_for_inference(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Prepare model for inference: eval mode, no grad, move to device.
    
    Args:
        model: The PyTorch model
        device: Target device
        
    Returns:
        The model, prepared for inference
    """
    model.eval()
    model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def prepare_for_training(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Prepare model for training: train mode, move to device.
    
    Args:
        model: The PyTorch model
        device: Target device
        
    Returns:
        The model, prepared for training
    """
    model.train()
    model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def get_memory_info(device: Optional[torch.device] = None) -> dict:
    """Get memory information for the device.
    
    Args:
        device: Target device. If None, uses current device.
        
    Returns:
        Dictionary with memory information
    """
    if device is None:
        device = get_device()
    
    info = {
        'device': device.type,
        'device_name': get_device_name(device),
    }
    
    if device.type == 'cuda':
        info.update({
            'allocated': torch.cuda.memory_allocated(device) / 1024**3,
            'reserved': torch.cuda.memory_reserved(device) / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated(device) / 1024**3,
        })
    elif device.type == 'mps':
        info['note'] = "MPS doesn't provide memory tracking like CUDA"
    
    return info


def clear_cache(device: Optional[torch.device] = None):
    """Clear GPU/MPS cache to free memory."""
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
