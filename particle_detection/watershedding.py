import random
from typing import Tuple, Collection, List, Iterable

import cv2
import mahotas
import matplotlib.cm
import matplotlib.colors
import numpy
from numpy import ndarray
from scipy.ndimage import morphology

from core import Particle


def _create_colormap() -> List[Tuple[float, float, float, float]]:
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
    return many_colors


COLOR_ARRAY = _create_colormap()
COLOR_MAP = matplotlib.colors.ListedColormap(COLOR_ARRAY, name="random", N=2000)


def distance_transform_to_labels(labels: ndarray, resolution: Tuple[float, float, float]):
    """Returns an image where pixels closer to the labels are brighter."""
    labels_inv = numpy.full_like(labels, 255, dtype=numpy.uint8)
    labels_inv[labels != 0] = 0
    distance_transform_to_labels = numpy.empty_like(labels, dtype=numpy.float64)
    distance_transform(labels_inv, distance_transform_to_labels, resolution)
    distance_transform_to_labels[distance_transform_to_labels > 4] = 4
    distance_transform_to_labels = 4 - distance_transform_to_labels
    return distance_transform_to_labels

def distance_transform(threshold: ndarray, out: ndarray, sampling: Tuple[float, float, float]):
    """Performs a 3D distance transform: all white pixels in the threshold are replaced by intensities representing the
    distance from black pixels.
    threshold: binary uint8 image
    out: float64 image of same size
    sampling: resolution of the image in arbitrary units in the z,y,x axis.
    """
    # noinspection PyTypeChecker
    morphology.distance_transform_edt(threshold, sampling=sampling, distances=out)


def watershed_maxima(threshold: ndarray, intensities: ndarray, minimal_size: Tuple[int, int, int]
                     ) -> Tuple[ndarray, ndarray]:
    """Performs a watershed transformation on the intensity image, using its regional maxima as seeds and flowing
    towards lower intensities."""
    kernel = numpy.ones(minimal_size)
    maxima = mahotas.morph.regmax(intensities, Bc=kernel)
    spots, n_spots = mahotas.label(maxima, Bc=kernel)
    surface = (intensities.max() - intensities)
    return watershed_labels(threshold, surface, spots, n_spots)


def create_labels(particles: Iterable[Particle], output: ndarray):
    """Creates a label image using the given labels. This image can be used for a watershed transform, for example.
    particles: list of particles, must have x/y/z in range of the output image
    output: integer image, which will contain the labels. label 1 == particle 0, label 2 == particle 1, etc."""
    i = 1
    for particle in particles:
        try:
            output[int(particle.z), int(particle.y), int(particle.x)] = i
            i += 1
        except IndexError:
            raise ValueError("The images do not match the cells: " + str(particle) + " is outside the image of size " +
                             str((output.shape[2], output.shape[1], output.shape[0])) + ".")


def watershed_labels(threshold: ndarray, surface: ndarray, label_image: ndarray, label_count: int) -> Tuple[ndarray, ndarray]:
    """Performs a watershed on the given surface, using the labels in label_image. The watershed will respect the
    threshold: Black (0) is background, rest is foreground. Black pixels in threshold are not watershedded.
    surface: float image, watershed goes from lower values to higher values
    label_image: 0 is unlabeled, 1, 2 etc are labels
    label_count: the amount of labels (excluding the background)
    return: Two images, one being a label image, the other being the watershed boundaries.
    """

    # Give areas outside the threshold a temporary label to avoid watershedding them
    new_label = label_count + 1
    label_image[threshold == 0] = new_label

    areas, lines = mahotas.cwatershed(surface, label_image, return_lines=True)
    areas[areas == new_label] = 0  # And remove the temporary label again
    areas[:, 0, 0] = areas.max()
    return areas, lines


# def remove_big_labels(labeled: ndarray):
#     """Removes all labels 10x larger than the average."""
#     sizes = mahotas.labeled.labeled_size(labeled)
#     too_big = numpy.where(sizes > numpy.median(sizes) * 10)
#     labeled[:] = mahotas.labeled.remove_regions(labeled, too_big, inplace=False)


