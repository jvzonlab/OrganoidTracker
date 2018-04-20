

import cv2
import skimage.feature
from matplotlib import pyplot
from numpy import ndarray

from particle_detection import Detector


class LocalMaximaDetector(Detector):
    """Blurs an image, then finds local maxima. Couldn't be simpler. However, it doesn't work: it detects way too many
    particles.
    """

    def detect(self, image: ndarray, show_results: bool = False, **kwargs):
        image_8bit = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)
        blurred = cv2.GaussianBlur(image_8bit, (0, 0), 7)
        local_maxima = skimage.feature.peak_local_max(blurred, min_distance=4)

        if show_results:
            pyplot.figure()
            pyplot.title("Original")
            pyplot.imshow(image_8bit, cmap="binary")
            pyplot.plot(local_maxima[:, 1], local_maxima[:, 0], 'o', color='red')
            pyplot.figure()
            pyplot.title("Blurred")
            pyplot.imshow(blurred)
            pyplot.show()
        else:
            return local_maxima