import random
from typing import Tuple

import mahotas

from numpy import ndarray
from scipy.ndimage import morphology
import numpy
import matplotlib.cm
import matplotlib.colors


def _create_colormap():
    many_colors = []
    many_colors += map(matplotlib.cm.get_cmap("prism"), range(256))
    many_colors += map(matplotlib.cm.get_cmap("cool"), range(256))
    many_colors += map(matplotlib.cm.get_cmap("summer"), range(256))
    many_colors += map(matplotlib.cm.get_cmap("spring"), range(256))
    many_colors += map(matplotlib.cm.get_cmap("autumn"), range(256))
    many_colors += map(matplotlib.cm.get_cmap("PiYG"), range(256))
    many_colors += map(matplotlib.cm.get_cmap("Spectral"), range(256))
    random.shuffle(many_colors)
    many_colors[0] = (0., 0., 0., 1.)  # We want a black background
    return matplotlib.colors.ListedColormap(many_colors, name="random", N=2000)


COLOR_MAP = _create_colormap()


def distance_transform(threshold: ndarray, out: ndarray, sampling: Tuple[float, float, float]):
    """Performs a 3D distance transform
    threshold: binary uint8 image
    out: float64 image of same size
    sampling: resolution of the image in arbitrary units in the z,y,x axis.
    """
    # noinspection PyTypeChecker
    morphology.distance_transform_edt(threshold, sampling=sampling, distances=out)


def watershed_maxima(threshold: ndarray, distances: ndarray, minimal_size: Tuple[int, int, int]) -> ndarray:
    kernel = numpy.ones(minimal_size)
    maxima = mahotas.morph.regmax(distances, Bc=kernel)
    spots, n_spots = mahotas.label(maxima, Bc=kernel)
    print("Found " + str(n_spots) + " particles")
    surface = (distances.max() - distances)
    areas: ndarray = mahotas.cwatershed(surface, spots)
    areas[threshold == 0] = 0
    areas[:, 0, 0] = areas.max()
    return areas

