import numpy
from numpy.core.multiarray import ndarray

from organoid_tracker.core.image_loader import ImageFilter


class ThresholdFilter(ImageFilter):

    _noise_limit: float  # Scaled 0 to 1

    def __init__(self, noise_limit: float = 0.08):
        """noise_limit = 0.5 would remove all pixels of less than 50% of the max intensity of the image."""
        self._noise_limit = noise_limit

    def filter(self, time_point, image_z, image: ndarray):
        image[image < self._noise_limit * image.max()] = 0

    def copy(self) -> ImageFilter:
        return ThresholdFilter(self._noise_limit)

    def get_name(self) -> str:
        return "Threshold"


class GaussianBlurFilter(ImageFilter):
    """Applies a Gaussian blur in 2D."""

    _blur_radius: int

    def __init__(self, blur_radius: int = 5):
        self._blur_radius = blur_radius

    def filter(self, time_point, image_z, image: ndarray):
        import cv2
        if len(image.shape) == 3:
            out = numpy.empty_like(image[0])
            for z in range(image.shape[0]):
                slice = image[z]
                cv2.GaussianBlur(slice, (self._blur_radius, self._blur_radius), 0, out)
                image[z] = out
        elif len(image.shape) == 2: # len(...) == 2
            out = numpy.empty_like(image)
            cv2.GaussianBlur(image, (self._blur_radius, self._blur_radius), 0, out)
            image[...] = out
        else:
            raise ValueError("Can only handle 2D or 3D images. Got shape " + str(image.shape))

    def copy(self) -> ImageFilter:
        return GaussianBlurFilter(self._blur_radius)

    def get_name(self) -> str:
        return "Gaussian blur"
