"""Attempt at edge detection. Doesn't work so well."""

import cv2

from numpy import ndarray
import numpy as np


# Load picture, convert to grayscale and detect edges
def perform(image: ndarray):
    gray = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    tight = cv2.Canny(blurred, 225, 250)

    kernel = np.ones((5,5),np.uint8)
    tight = cv2.dilate(tight, kernel)
    tight = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, kernel)

    # show the images
    cv2.imshow("Edges",tight)
    cv2.waitKey(0)
