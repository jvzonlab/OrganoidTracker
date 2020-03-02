"""Code for fitting cells to Gaussian functions."""
import sys
from timeit import default_timer
from typing import List, Iterable, Dict, Optional

import cv2
import mahotas
import numpy
import scipy.optimize
from matplotlib import cm, pyplot
from numpy import ndarray
from tifffile import tifffile

from organoid_tracker.core.bounding_box import bounding_box_from_mahotas, BoundingBox
from organoid_tracker.core.gaussian import Gaussian
from organoid_tracker.core.images import Image
from organoid_tracker.core.mask import create_mask_for
from organoid_tracker.core.position import Position
from organoid_tracker.position_detection import smoothing, clusterer
from organoid_tracker.util.mpl_helper import QUALITATIVE_COLORMAP
from organoid_tracker.visualizer.debug_image_visualizer import popup_3d_image


class _ModelAndImageDifference:
    _data_image: ndarray

    # Some reusable images (to avoid allocating large new arrays)
    _scratch_image: ndarray  # Used for drawing the Gaussians
    _scratch_image_gradient: ndarray  # Used for drawing dG/dv, with v a parameters of a Gaussian

    _last_gaussians: Dict[Gaussian, ndarray]

    def __init__(self, data_image: ndarray):
        self._data_image = data_image.astype(numpy.float64)
        self._scratch_image = numpy.empty_like(self._data_image)
        self._scratch_image_gradient = numpy.empty_like(self._data_image)
        self._last_gaussians = dict()

    def difference_with_image(self, params: ndarray) -> float:
        if not self._draw_gaussians_to_scratch_image(params):
            return sys.float_info.max

        self._scratch_image -= self._data_image
        self._scratch_image **= 2
        sum = self._scratch_image.sum()
        return sum

    def _draw_gaussians_to_scratch_image(self, params: ndarray) -> bool:
        """Makes self._scratch_image equal to ∑g(x). Returns False if mathematically impossible parameters are given."""
        self._scratch_image.fill(0)
        last_gaussians_new = dict()
        for i in range(0, len(params), 10):
            gaussian_params = params[i:i + 10]
            try:
                gaussian = Gaussian(*gaussian_params)
            except ValueError:
                return False
            else:
                cached_image = self._last_gaussians.get(gaussian)
                last_gaussians_new[gaussian] = gaussian.draw(self._scratch_image, cached_image)
        self._last_gaussians = last_gaussians_new
        return True

    def gradient(self, params: ndarray) -> ndarray:
        """Calculates the gradient of self.difference_with_image for all of the possible parameters."""
        # Calculate 2 * (−I(x) + ∑g(x))
        self._draw_gaussians_to_scratch_image(params)
        self._scratch_image -= self._data_image
        self._scratch_image *= 2

        # Multiply with one of the derivatives of one of the gradients
        gradient_for_each_parameter = numpy.empty_like(params)
        for gaussian_index in range(len(params) // 10):
            param_pos = gaussian_index * 10
            gaussian = Gaussian(*params[param_pos:param_pos + 10])
            for gradient_nr in range(10):
                # Every param gets its own gradient
                self._scratch_image_gradient.fill(0)
                gaussian.draw_gradient(self._scratch_image_gradient, gradient_nr)
                self._scratch_image_gradient *= self._scratch_image
                gradient_for_each_parameter[gaussian_index * 10 + gradient_nr] = self._scratch_image_gradient.sum()
        return gradient_for_each_parameter


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


def perform_gaussian_mixture_fit(original_image: ndarray, guesses: List[Gaussian]) -> List[Gaussian]:
    """Fits multiple Gaussians to the image (a Gaussian Mixture Model). Initial seeds must be given."""
    if len(guesses) > 5:
        raise ValueError(f"Minimization failed: too many parameters (tried to fit {len(guesses)} Gaussian functions)")

    model_and_image_difference = _ModelAndImageDifference(original_image)

    guesses_list = []
    for guess in guesses:
        guesses_list += guess.to_list()

    result = scipy.optimize.minimize(model_and_image_difference.difference_with_image, guesses_list,
    #                                 method='BFGS', jac=model_and_image_difference.gradient, options={'gtol': 2000})
    #                                method = 'Newton-CG', jac = model_and_image_difference.gradient, options = {'disp': True, 'xtol': 0.1})
                                     method='Powell', options = {'ftol': 0.001, 'xtol': 10})
    #                                method="Nelder-Mead", options = {'fatol': 0.1, 'xtol': 0.1, 'adaptive': False, 'disp': True})

    if not result.success:
        raise ValueError("Minimization failed: " + result.message)

    result_gaussians = []
    for i in range(0, len(result.x), 10):
        gaussian_params = result.x[i:i + 10]
        result_gaussians.append(Gaussian(*gaussian_params))
    return result_gaussians


_FIT_MARGIN = 5


def perform_gaussian_mixture_fit_from_watershed(image: ndarray, watershed_image: ndarray, positions: List[Position],
                                                blur_radius: int, erode_passes: int) -> List[Gaussian]:
    """GMM using watershed as seeds. The watershed is used to fit as few Gaussians at the same time as possible: if two
    colors in the watershed have only a small connection (defined by erode_passes) they will be fit separately. The
    positions are used as starting positions for the Gaussian fit; the index in the list must match the index in the
    watershed image."""
    start_time = default_timer()

    # Find out where the positions are
    bounding_boxes = mahotas.labeled.bbox(watershed_image.astype(numpy.int32))

    # Using a watershed to check which cell overlap
    clusters, cluster_image = clusterer.get_clusters_from_labeled_image(watershed_image, positions, erode_passes)

    all_gaussians: List[Optional[Gaussian]] = [None] * len(positions)  # Initialize empty list

    for cluster in clusters:
        # To keep the fitting procedure easy, we try to fit as few cells at the same time as possible
        # Only overlapping nuclei should be fit together. Overlap was detected using a watershed, see clusterer above.
        cell_ids = cluster.get_tags()
        print(f"Fitting {len(cell_ids)} cells: " + str([positions[cell_id] for cell_id in cell_ids]))
        bounding_box = _merge_bounding_boxes(bounding_boxes, cell_ids)
        bounding_box.expand(x=blur_radius, y=blur_radius, z=0)

        mask = create_mask_for(Image(image))
        mask.set_bounds(bounding_box)
        if mask.has_zero_volume():
            continue

        gaussians = []
        for cell_id in cell_ids:
            center = positions[cell_id]
            if center is None:
                print("No position for cell " + str(center))
                continue  # No center of mass for this cell id
            intensity = image[int(center.z), int(center.y), int(center.x)]

            mask.add_from_labeled(watershed_image, cell_id)

            gaussians.append(Gaussian(intensity, center.x, center.y, center.z, 50, 50, 2, 0, 0, 0))
        mask.dilate_xy(blur_radius // 2)
        cropped_image = mask.create_masked_image(Image(image))
        cropped_image = _add_border(cropped_image, _FIT_MARGIN)
        smoothing.smooth(cropped_image, blur_radius)

        offset_x, offset_y, offset_z = bounding_box.min_x - _FIT_MARGIN, bounding_box.min_y - _FIT_MARGIN, bounding_box.min_z - _FIT_MARGIN
        gaussians = [gaussian.translated(-offset_x, -offset_y, -offset_z) for gaussian in gaussians]
        try:
            gaussians = perform_gaussian_mixture_fit(cropped_image, gaussians)
            for i, gaussian in enumerate(gaussians):
                all_gaussians[cell_ids[i]] = gaussian.translated(offset_x, offset_y, offset_z)
        except ValueError:
            print("Minimization failed for " + str(cluster))
            continue
    end_time = default_timer()
    print("Whole fitting process took " + str(end_time - start_time) + " seconds.")
    all_gaussians = all_gaussians[1:]  # Remove first element, that's the background of the image
    return all_gaussians


def _add_border(array: ndarray, pixels: int) -> ndarray:
    new_array = numpy.zeros((array.shape[0] + 2 * pixels, array.shape[1] + 2 * pixels, array.shape[2] + 2 * pixels),
                            dtype=array.dtype)
    new_array[pixels:array.shape[0] + pixels, pixels:array.shape[1] + pixels, pixels:array.shape[2] + pixels] = array
    return new_array


def _merge_bounding_boxes(all_boxes: ndarray, cell_ids: Iterable[int]) -> BoundingBox:
    """Creates a bounding box object that encompasses the bounding boxes of all the given cells."""
    combined_bounding_box = None
    for cell_id in cell_ids:
        if cell_id >= len(all_boxes):
            continue  # No bounding box of this cell id
        bounding_box = all_boxes[cell_id]
        if combined_bounding_box is None:
            combined_bounding_box = bounding_box
            continue
        combined_bounding_box[0] = min(combined_bounding_box[0], bounding_box[0])
        combined_bounding_box[1] = max(combined_bounding_box[1], bounding_box[1])
        combined_bounding_box[2] = min(combined_bounding_box[2], bounding_box[2])
        combined_bounding_box[3] = max(combined_bounding_box[3], bounding_box[3])
        combined_bounding_box[4] = min(combined_bounding_box[4], bounding_box[4])
        combined_bounding_box[5] = max(combined_bounding_box[5], bounding_box[5])
    if combined_bounding_box is None:
        return BoundingBox(0, 0, 0, 0, 0, 0)
    return bounding_box_from_mahotas(combined_bounding_box)


def _dilate(image_3d: ndarray):
    scratch_2d = numpy.empty_like(image_3d[0])
    kernel = numpy.ones((5, 5), numpy.uint8)
    for z in range(image_3d.shape[0]):
        cv2.dilate(image_3d[z], kernel, dst=scratch_2d, iterations=2)
        image_3d[z] = scratch_2d


