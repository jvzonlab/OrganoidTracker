import cv2
import numpy as np
import skimage.feature
from numpy import ndarray
from matplotlib import pyplot


def perform(image: ndarray):
    cvuint8 = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)
    ret, thresh = cv2.threshold(cvuint8, 90, 256, cv2.THRESH_BINARY)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    blur = cv2.GaussianBlur(dist_transform, (0, 0), 4)
    local_maxima = skimage.feature.peak_local_max(blur, min_distance=4)

    pyplot.figure()
    pyplot.title("Original")
    pyplot.imshow(cvuint8, cmap="binary")
    pyplot.plot(local_maxima[:, 1], local_maxima[:, 0], 'o', color='red')
    pyplot.figure()
    pyplot.title("Thresholded")
    pyplot.imshow(thresh)
    pyplot.figure()
    pyplot.title("Distance transform")
    pyplot.imshow(dist_transform)
    pyplot.figure()
    pyplot.title("Blurred distance transform")
    pyplot.imshow(blur)
    pyplot.show()