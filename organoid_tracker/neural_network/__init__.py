from typing import Union
import numpy

Tensor = "torch.Tensor"  # Now pointing to PyTorch, but if we ever switch again we can change this

TensorLike = Union[Tensor, int, float, numpy.ndarray]

# Default target resolution for position models
# New models should always specify their target resolution in settings.json
DEFAULT_TARGET_RESOLUTION_ZYX_UM = (2.0, 0.32, 0.32)
