from numpy import ndarray
import numpy
import cv2


def smooth(image_stack: ndarray, smooth_size: int):
    """Smooths the image, z-plane for z-plane, using a Gaussian kernel of `smooth_size * smooth_size` pixels. THe
    input image is overwritten."""
    temp = numpy.empty_like(image_stack[0])
    for z in range(image_stack.shape[0]):
        cv2.GaussianBlur(image_stack[z], (smooth_size, smooth_size), 0, dst=temp)
        image_stack[z] = temp


def get_smoothed(image_stack: ndarray, smooth_size: int) -> ndarray:
    """SSmooths the image, z-plane for z-plane, using a Gaussian kernel of `smooth_size * smooth_size` pixels. The
    result is returned, the input image is left untouched."""
    out = numpy.empty_like(image_stack)
    for z in range(image_stack.shape[0]):
        cv2.GaussianBlur(image_stack[z], (smooth_size, smooth_size), 0, dst=out[z])
    return out