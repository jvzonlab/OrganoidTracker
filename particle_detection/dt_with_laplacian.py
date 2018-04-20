"""Image derivative visualization"""

import cv2
import matplotlib.pyplot as plt
import numpy
import skimage.feature
from matplotlib.figure import Figure
from numpy import ndarray

from gui import dialog
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
            def paint_figure(figure: Figure):
                ((ax1, ax2), (ax3, ax4)) = figure.subplots(2, 2)
                ax1.title.set_text("Original image")
                ax1.imshow(image_8bit, cmap="binary")
                ax1.plot(local_maxima[:, 1], local_maxima[:, 0], 'o', color='red', markersize=2)

                ax2.title.set_text("Laplacian")
                image = ax2.imshow(gaussian_laplacian, cmap="Spectral")
                figure.colorbar(image, ax=ax2)

                ax3.title.set_text("Threshold sum")
                ax3.imshow(thresholded_sum, cmap="binary")

            dialog.popup_figure(paint_figure)

        return local_maxima

def _new_figure():
    plt.figure()
    plt.tight_layout()
    plt.axis('off')