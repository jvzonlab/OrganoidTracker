import cv2
import numpy
from numpy import ndarray


def image_to_8bit(image: ndarray):
    """Converts and scales the image to an 8-bit image, such that the highest value in the image becomes 255. So even if the image
    is already an 8-bit image, it can still get rescaled.

    Note that this method returns a copy and does not modify the original image."""
    return cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)


def ensure_8bit(image: ndarray):
    """Converts an image to uint8 if it isn't already."""
    if image.dtype == numpy.uint8:
        return image
    return image_to_8bit(image)
