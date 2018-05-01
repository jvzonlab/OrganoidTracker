import random

import mahotas

from numpy import ndarray
from scipy.ndimage import morphology
import cv2
import numpy
import matplotlib.cm
import matplotlib.colors


def _create_colormap():
    many_colors = []
    many_colors+= map(matplotlib.cm.get_cmap("prism"), range(256))  # Get all colors of Prism
    random.shuffle(many_colors)
    many_colors[0] = (0., 0., 0., 1.)  # We want a black background
    return matplotlib.colors.ListedColormap(many_colors, name="random", N=1024)


COLOR_MAP = _create_colormap()


def distance_transform(threshold: ndarray, out: ndarray):
    """Threshold: binary uint8 image, out: float64 image of same size."""
    # noinspection PyTypeChecker
    morphology.distance_transform_edt(threshold, sampling=(2, 0.32, 0.32), distances=out)

    temp = numpy.empty_like(out[0])
    for z in range(out.shape[0]):
        cv2.GaussianBlur(out[z], (5, 5), 0, dst=temp)
        out[z] = temp


def watershed_maxima(threshold: ndarray, distances: ndarray) -> ndarray:
    kernel = numpy.ones((3, 9, 9))
    maxima = mahotas.morph.regmax(distances, Bc=kernel)
    spots, n_spots = mahotas.label(maxima, Bc=kernel)
    print("Found " + str(n_spots) + " spots")
    surface = (distances.max() - distances)
    areas: ndarray = mahotas.cwatershed(surface, spots)
    print("Type of areas: " + str(areas.dtype))
    areas[threshold == 0] = 0
    areas[:, 0, 0] = areas.max()
    return areas

