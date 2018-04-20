

import cv2
import numpy as np
import skimage.feature
from matplotlib import pyplot
from numpy import ndarray

from particle_detection import Detector


class DistanceTransformDetector(Detector):
    """The original distance transform detection. A black-and-white image is created by thresholding. Then a
    distance transform is applied. The local maxima are then considered to be the particles."""

    def detect(self, image: ndarray, min_intensity: float = 0.6, show_results: bool = False, **kwargs):
        cvuint8 = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)
        ret, thresh = cv2.threshold(cvuint8, int(min_intensity * 255), 255, cv2.THRESH_BINARY)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        blur = cv2.GaussianBlur(dist_transform, (0, 0), 4)
        local_maxima = skimage.feature.peak_local_max(blur, min_distance=4)

        if show_results:
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
        else:
            return local_maxima