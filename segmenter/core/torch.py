import torch


def get_default_device_type() -> str:
    """
    Pick GPU if available, else CPU
    Chooses MPS for Apple MPS devices, or CUDA device if available
    """
    # _device = "cpu"
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"  # For Apple devices with MPS support
    else:
        _device = "cpu"
    return _device


def get_default_device() -> torch.device:
    return torch.device(get_default_device_type())


def set_default_device(device: torch.device):
    if device.type == "cuda":
        torch.set_default_dtype(torch.float16)
    elif device.type == "mps" or device.type == "cpu":
        torch.set_default_dtype(torch.float)

    if torch.amp.autocast_mode.is_autocast_available(device.type):
        torch.autocast(device.type,
                       dtype=torch.float).__enter__()
    return


def report_cuda_memory_usage(device: torch.device, label=None) -> str:
    if device.type.startswith('cuda'):
        out_str = label if label is not None else ""
        out_str += f"\nMemory Usage:\n{torch.cuda.memory_usage(device=device)}\n"

        out_str += f"Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB\n"
        out_str += f"Reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB\n"
    else:
        out_str = f"\nMDevice {device} is not a CUDA device\n"
    return out_str

