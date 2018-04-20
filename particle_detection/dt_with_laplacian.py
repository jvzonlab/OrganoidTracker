"""Image derivative visualization"""

import cv2
import matplotlib.pyplot as plt
import numpy
import skimage.feature
from numpy import ndarray

from particle_detection import Detector


class DistanceTransformWithLaplacianDetector(Detector):
    """Similar to DistanceTransformDetection, but with a threshold of the laplacian of the image added. This is done
    so that particles with only low intensities are also detected. The threshold of the image itself is still used, as
    that provides better information of the shape of a particle than the laplacian. (The laplacian tends to empathize
    the corners of a particle.)
    """

    def detect(self, image: ndarray, show_results = False, min_intensity = 0.6, max_laplacian = -1, **kwargs) -> ndarray:
        image_8bit = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)

        # Create threshold from laplacian
        gaussian = cv2.GaussianBlur(image_8bit,(0,0),5)
        laplacian = cv2.Laplacian(gaussian,cv2.CV_32F)
        gaussian_laplacian = cv2.GaussianBlur(laplacian,(0,0),2)
        gaussian_laplacian *= -1
        ret, thresholded_laplacian = cv2.threshold(gaussian_laplacian, -max_laplacian, 255, cv2.THRESH_BINARY)
        kernel = numpy.ones((3, 3), numpy.uint8)
        thresholded_laplacian = numpy.uint8(thresholded_laplacian)

        ret, thresholded_image = cv2.threshold(image_8bit, int(min_intensity * 255), 255, cv2.THRESH_BINARY)
        thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)

        thresholded_sum = cv2.bitwise_or(thresholded_laplacian, thresholded_image)

        dist_transform = cv2.distanceTransform(thresholded_sum, cv2.DIST_L2, 5)
        dist_transform_blurred = cv2.GaussianBlur(dist_transform, (0, 0), 4)
        local_maxima = skimage.feature.peak_local_max(dist_transform_blurred, min_distance=4)

        # show the images
        if show_results:
            _new_figure()
            plt.title("Original image")
            plt.imshow(image_8bit, cmap="binary")
            plt.plot(local_maxima[:, 1], local_maxima[:, 0], 'o', color='red')
            plt.colorbar()
            _new_figure()
            plt.title("Laplacian")
            plt.imshow(gaussian_laplacian, cmap="Spectral")
            plt.colorbar()
            _new_figure()
            plt.title("Threshold applied on laplacian")
            plt.imshow(thresholded_laplacian, cmap="binary")
            _new_figure()
            plt.title("Threshold applied on image")
            plt.imshow(thresholded_image, cmap="binary")
            _new_figure()
            plt.title("Threshold sum")
            plt.imshow(thresholded_sum, cmap="binary")
            plt.show()

        return local_maxima

def _new_figure():
    plt.figure()
    plt.tight_layout()
    plt.axis('off')