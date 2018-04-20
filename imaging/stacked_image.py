from typing import Optional

from numpy import ndarray
import numpy

from core import ImageLoader, TimePoint


def average_out(image_stack: ndarray, num_layers=2):
    output = numpy.zeros_like(image_stack, dtype=numpy.float32)

    height = image_stack.shape[0]
    for i in range(height):
        start_z = max(0, i - num_layers)
        end_z = min(height, i + num_layers)
        slice = image_stack[start_z: end_z]
        numpy.mean(slice, axis=0, out=output[i], dtype=numpy.float32)

    return output


class AveragingImageLoader(ImageLoader):
    _internal: ImageLoader

    def __init__(self, image_loader: ImageLoader):
        self._internal = image_loader

    def load_3d_image(self, time_point: TimePoint) -> Optional[ndarray]:
        """Loads an image, usually from disk. Returns None if there is no image for this time point."""
        return average_out(self._internal.load_3d_image(time_point))

    def unwrap(self) -> ImageLoader:
        return self._internal
