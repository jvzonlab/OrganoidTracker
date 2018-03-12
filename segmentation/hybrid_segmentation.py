"""Segmentation using both intensities and the laplacian."""
from numpy import ndarray
import cv2
import numpy
import math


def perform(image: ndarray) -> ndarray:
    """Returns a black-and-white image where white is particle and black is background, at least in theory."""
    image_8bit = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)

    # Crop to a circle
    image_circle = numpy.zeros_like(image_8bit)
    width = image_circle.shape[1]
    height = image_circle.shape[0]
    circle_size = min(width, height) - 3
    cv2.circle(image_circle, (int(width/2), int(height/2)), int(circle_size/2), 255, -1)
    cv2.bitwise_and(image_8bit, image_circle, image_8bit)

    ret, thresholded_image = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_image, contours, hierarchy = cv2.findContours(thresholded_image, 1, 2)

    for i in range(len(contours)):
        contour_image = cv2.drawContours(contour_image, contours, i, 60 + i * 31, 1)
        perimeter = cv2.arcLength(contours[i], True)
        area = cv2.contourArea(contours[i])
        area_perimeter = 4 * math.pi * area / perimeter**2 if perimeter > 0 else 0
        print("Perimeter: " + str(perimeter) + "  Area: " + str(area) + "  Area/perimeter^2: " + str(area_perimeter))
    print("---")
    return contour_image
