"""Attempt at edge detection. Doesn't work so well."""

import cv2
import numpy as np
from numpy import ndarray


# Load picture, convert to grayscale and detect edges
def perform(image: ndarray):
    gray = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    tight = cv2.Canny(blurred, 225, 250)

    kernel = np.ones((5,5),np.uint8)

    # show the images
    cv2.imshow("Edges",tight)
    cv2.waitKey(0)
