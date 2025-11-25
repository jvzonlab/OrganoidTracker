from typing import Union
import numpy

Tensor = "torch.Tensor"  # Now pointing to PyTorch, but if we ever switch again we can change this

TensorLike = Union[Tensor, int, float, numpy.ndarray]
