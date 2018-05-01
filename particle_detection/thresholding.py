"""Attempt at edge detection. Doesn't work so well."""

import cv2
import numpy
from numpy import ndarray
from segmentation import iso_intensity_curvature
from segmentation.iso_intensity_curvature import ImageDerivatives


def image_to_8bit(image: ndarray):
    return cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)


def background_removal(orignal_image_8bit: ndarray, threshold: ndarray):
    """Sets all elements below 10% of 255 to zero."""
    absolute_thresh = int(0.1 * 255)
    threshold[orignal_image_8bit < absolute_thresh] = 0


def adaptive_threshold(image_8bit: ndarray, out: ndarray, block_size=51):
    """A simple, adaptive threshold. Intensities below 10% are removed, as well as intensities that fall below an
    adaptive Gaussian threshold.
    """
    for z in range(image_8bit.shape[0]):
        cv2.adaptiveThreshold(image_8bit[z], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size,
                              2, dst=out[z])
    background_removal(image_8bit, out)


def advanced_threshold(image_8bit: ndarray, out: ndarray):
    adaptive_threshold(image_8bit, out)

    curvature_out = numpy.full_like(image_8bit, 255, dtype=numpy.uint8)
    iso_intensity_curvature.get_negative_gaussian_curvatures(image_8bit, ImageDerivatives(), curvature_out)
    out &= curvature_out

    fill_threshold(out)
    background_removal(image_8bit, out)
    erode_threshold(out)


def open_threshold(threshold: ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    temp = numpy.empty_like(threshold[0])
    for z in range(threshold.shape[0]):
        cv2.morphologyEx(threshold[z], cv2.MORPH_OPEN, kernel, dst=temp)
        threshold[z] = temp


def dilate_threshold(threshold: ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    temp = numpy.empty_like(threshold[0])
    for z in range(threshold.shape[0]):
        cv2.dilate(threshold[z], kernel, dst=temp)
        threshold[z] = temp


def erode_threshold(threshold: ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = numpy.empty_like(threshold[0])
    for z in range(threshold.shape[0]):
        cv2.erode(threshold[z], kernel, dst=temp, iterations=1)
        threshold[z] = temp


def close_threshold(threshold: ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    temp = numpy.empty_like(threshold[0])
    for z in range(threshold.shape[0]):
        cv2.morphologyEx(threshold[z], cv2.MORPH_CLOSE, kernel, dst=temp)
        threshold[z] = temp


def fill_threshold(threshold: ndarray):
    temp = numpy.empty_like(threshold[0])
    for z in range(threshold.shape[0]):
        _fill_threshold_2d(threshold[z], temp)


def _fill_threshold_2d(threshold: ndarray, temp_for_floodfill: ndarray):
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    temp_for_floodfill[:] = threshold

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = threshold.shape[:2]
    mask = numpy.zeros((h + 2, w + 2), numpy.uint8)

    # Floodfill from point (0, 0) = top left
    cv2.floodFill(temp_for_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image, combine with threshold
    im_floodfill_inv = cv2.bitwise_not(temp_for_floodfill)
    threshold |= im_floodfill_inv