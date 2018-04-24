"""Segmentation using both intensities and the laplacian."""
import math

import cv2
import numpy
from matplotlib.axis import Axis
from matplotlib.patches import Ellipse
from numpy import ndarray
from typing import Tuple, List


def perform(image: ndarray) -> Tuple[ndarray, List[Ellipse]]:
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
    thresholded_image = cv2.erode(thresholded_image, kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    contour_image, contours, hierarchy = cv2.findContours(thresholded_image, 1, 2)

    ellipses = []
    for i in range(len(contours)):
        contour_image = cv2.drawContours(contour_image, contours, i, 60 + i * 31, 1)
        perimeter = cv2.arcLength(contours[i], True)
        area = cv2.contourArea(contours[i])
        area_perimeter = 4 * math.pi * area / perimeter**2 if perimeter > 0 else 0

        print("Perimeter: " + str(perimeter) + "  Area: " + str(area) + "  Area/perimeter^2: " + str(area_perimeter))
        if perimeter > 15:
            ellipse = cv2.fitEllipse(contours[i])
            print("Ellipse: " + str(ellipse))
            ellipses.append(Ellipse(xy=ellipse[0], width=ellipse[1][0], height=ellipse[1][1], angle=ellipse[2],
                                      fill=False, linewidth=8, linestyle='dashed', edgecolor="white"))
        else:  # Too few pixels to draw reliable perimeter
            print("No ellipse")
    print("---")

    return contour_image, ellipses