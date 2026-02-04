"""
Model wrappers for non-sklearn backends.
"""

import config


def get_torch_device():
    """
    Get the appropriate torch device based on config.USE_GPU setting.
    """
    import torch

    use_gpu = getattr(config, "USE_GPU", None)

    if use_gpu is False:
        return torch.device("cpu")
    if use_gpu is True:
        if torch.cuda.is_available():
            return torch.device("cuda")
        import warnings
        warnings.warn("USE_GPU=True but CUDA not available, falling back to CPU")
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accelerator_for_lightning():
    """
    Get accelerator string for PyTorch Lightning trainer.
    Respects config.USE_GPU setting.
    """
    import torch

    use_gpu = getattr(config, "USE_GPU", None)

    if use_gpu is False:
        return "cpu"
    if use_gpu is True:
        return "gpu" if torch.cuda.is_available() else "cpu"

    return "gpu" if torch.cuda.is_available() else "cpu"
