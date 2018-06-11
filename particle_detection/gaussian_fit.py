from timeit import default_timer as timer
from typing import Tuple, List, Iterable, Dict, Optional

import cv2
import networkx
import numpy
import scipy.optimize
from networkx import Graph
from numpy import ndarray

from core import Particle
from particle_detection import smoothing
from particle_detection.ellipse import Ellipse, EllipseStack, EllipseCluster

from core.gaussian import Gaussian

ELLIPSE_SHRINK_PIXELS = 2


class _ModelAndImageDifference:
    _data_image: ndarray
    _scratch_image: ndarray
    _last_gaussians: Dict[Gaussian, ndarray]

    def __init__(self, data_image: ndarray):
        self._data_image = data_image.astype(numpy.float64)
        self._scratch_image = numpy.empty_like(self._data_image)
        self._last_gaussians = dict()

    def difference_with_image(self, params) -> float:
        last_gaussians_new = dict()

        self._scratch_image.fill(0)
        for i in range(0, len(params), 10):
            gaussian_params = params[i:i + 10]
            gaussian = Gaussian(*gaussian_params)
            cached_image = self._last_gaussians.get(gaussian)
            last_gaussians_new[gaussian] = gaussian.draw(self._scratch_image, cached_image)
        self._last_gaussians = last_gaussians_new

        self._scratch_image -= self._data_image
        self._scratch_image **= 2
        sum = self._scratch_image.sum()
        return sum


def add_noise(data: ndarray):
    """Adds noise to the given data. Useful for construction of artificial testing data."""
    shape = data.shape
    numpy.random.seed(1949)  # Make sure noise is reproducible
    data = data.ravel()
    data += 20 * numpy.random.normal(size=len(data))
    return data.reshape(*shape)


def perform_gaussian_fit(original_image: ndarray, guess: Gaussian) -> Gaussian:
    """Fits a gaussian function to an image. original_image is a zyx-indexed image, guess is an initial starting point
    for the fit."""
    return perform_gaussian_mixture_fit(original_image, [guess])[0]


def perform_gaussian_mixture_fit(original_image: ndarray, guesses: Iterable[Gaussian]) -> List[Gaussian]:
    """Fits multiple Gaussians to the image (a Gaussian Mixture Model). Initial seeds must be given."""
    model_and_image_difference = _ModelAndImageDifference(original_image)

    guesses_list = []
    for guess in guesses:
        guesses_list += guess.to_list()

    result = scipy.optimize.minimize(model_and_image_difference.difference_with_image, guesses_list,
                                     method='Powell', options={'ftol':0.001,'xtol':10})
    if not result.success:
        raise ValueError("Minimization failed: " + result.message)

    result_gaussians = []
    for i in range(0, len(result.x), 10):
        gaussian_params = result.x[i:i + 10]
        result_gaussians.append(Gaussian(*gaussian_params))
    return result_gaussians


def perform_gaussian_mixture_fit_from_watershed(image: ndarray, watershed_image: ndarray, blur_radius: int
                                                ) -> List[Gaussian]:
    """GMM using watershed as seeds. The watershed is used to fit as few Gaussians at the same time as possible."""
    ellipse_stacks = _get_ellipse_stacks(watershed_image)
    ellipse_clusters = _get_overlapping_stacks(ellipse_stacks)

    all_gaussians: List[Optional[Gaussian]] = [None] * len(ellipse_stacks)  # Initialize empty list
    start_time = timer()
    for cluster in ellipse_clusters:
        offset_x, offset_y, offset_z, cropped_image = cluster.get_image_for_fit(image, blur_radius)
        if cropped_image is None:
            continue
        smoothing.smooth(cropped_image, blur_radius)
        gaussians = cluster.guess_gaussians(image)
        tags = cluster.get_tags()

        gaussians = [gaussian.translated(-offset_x, -offset_y, -offset_z) for gaussian in gaussians]
        gaussians = perform_gaussian_mixture_fit(cropped_image, gaussians)
        for i, gaussian in enumerate(gaussians):
            all_gaussians[tags[i]] = gaussian.translated(offset_x, offset_y, offset_z)
    end_time = timer()
    print("Whole fitting process took " + str(end_time - start_time) + " seconds.")
    return all_gaussians


def _dilate(image_3d: ndarray):
    scratch_2d = numpy.empty_like(image_3d[0])
    kernel = numpy.ones((5, 5), numpy.uint8)
    for z in range(image_3d.shape[0]):
        cv2.dilate(image_3d[z], kernel, dst=scratch_2d, iterations=2)
        image_3d[z] = scratch_2d


def _get_ellipse_stacks(watershed: ndarray) -> List[EllipseStack]:
    """Gets ellipse stacks from a watershed image. The stack tagged as nr. 0 is constructed from label 1, stack 1 from
    label 2, etc."""
    buffer = numpy.empty_like(watershed, dtype=numpy.uint8)
    ellipse_stacks = []
    for i in range(1, watershed.max() + 1):   # + 1 to ensure inclusive range
        ellipse_stack = []
        buffer.fill(0)
        buffer[watershed == i] = 255
        for z in range(buffer.shape[0]):
            contour_image, contours, hierarchy = cv2.findContours(buffer[z], cv2.RETR_LIST, 2)
            contour_index, area = _find_contour_with_largest_area(contours)
            if contour_index == -1 or len(contours[contour_index]) < 5:
                ellipse_stack.append(None)
                continue  # No contours found
            ellipse_pos, ellipse_size, ellipse_angle = cv2.fitEllipse(contours[contour_index])
            if ellipse_size[0] <= ELLIPSE_SHRINK_PIXELS or ellipse_size[1] <= ELLIPSE_SHRINK_PIXELS:
                ellipse_stack.append(None)
                continue  # Ellipse is too small
            ellipse_stack.append(Ellipse(ellipse_pos[0], ellipse_pos[1], ellipse_size[0] - ELLIPSE_SHRINK_PIXELS,
                                         ellipse_size[1] - ELLIPSE_SHRINK_PIXELS, ellipse_angle))
        ellipse_stacks.append(EllipseStack(ellipse_stack, tag=i - 1))
    return ellipse_stacks


def _get_overlapping_stacks(stacks: List[EllipseStack]) -> List[EllipseCluster]:
    cell_network = Graph()
    for stack in stacks:
        cell_network.add_node(stack)
        for other_stack in stacks:
            if other_stack is stack:
                continue  # Ignore self-overlapping
            if other_stack not in cell_network:
                continue  # To be processed later
            if stack.intersects(other_stack):
                cell_network.add_edge(stack, other_stack)

    clusters = []
    for cluster in networkx.connected_components(cell_network):
        clusters.append(EllipseCluster(cluster))
    return clusters


def _find_contour_with_largest_area(contours) -> Tuple[int, float]:
    highest_area = 0
    index_with_highest_area = -1
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > highest_area:
            highest_area = area
            index_with_highest_area = i
    return index_with_highest_area, highest_area
