"""Code for fitting cells to Gaussian functions."""

from timeit import default_timer
from typing import List, Iterable, Dict, Optional

import cv2
import mahotas
import numpy
import scipy.optimize
from numpy import ndarray

from ai_track.core.bounding_box import bounding_box_from_mahotas, BoundingBox
from ai_track.core.gaussian import Gaussian
from ai_track.core.images import Image
from ai_track.core.mask import create_mask_for
from ai_track.position_detection import smoothing, ellipse_cluster


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
        self._draw_gaussians_to_scratch_image(params)

        self._scratch_image -= self._data_image
        self._scratch_image **= 2
        sum = self._scratch_image.sum()
        return sum

    def _draw_gaussians_to_scratch_image(self, params: ndarray):
        # Makes self._scratch_image equal to ∑g(x)
        self._scratch_image.fill(0)
        last_gaussians_new = dict()
        for i in range(0, len(params), 10):
            gaussian_params = params[i:i + 10]
            gaussian = Gaussian(*gaussian_params)
            cached_image = self._last_gaussians.get(gaussian)
            last_gaussians_new[gaussian] = gaussian.draw(self._scratch_image, cached_image)
        self._last_gaussians = last_gaussians_new

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
    if len(guesses) > 30:
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


def perform_gaussian_mixture_fit_from_watershed(image: ndarray, watershed_image: ndarray, blur_radius: int) -> List[Gaussian]:
    """GMM using watershed as seeds. The watershed is used to fit as few Gaussians at the same time as possible."""
    start_time = default_timer()

    # Using ellipses to check which cell overlap
    ellipse_stacks = ellipse_cluster.get_ellipse_stacks_from_watershed(watershed_image)
    clusters = ellipse_cluster.find_overlapping_stacks(ellipse_stacks)

    # Find out where the positions are
    bounding_boxes = mahotas.labeled.bbox(watershed_image.astype(numpy.int32))
    position_centers = mahotas.center_of_mass(image, watershed_image)

    all_gaussians: List[Optional[Gaussian]] = [None] * len(ellipse_stacks)  # Initialize empty list

    for cluster in clusters:
        # To keep the fitting procedure easy, we try to fit as few cells at the same time as possible
        # Only overlapping nuclei should be fit together. Overlap is detected using ellipses.
        cell_ids = cluster.get_tags()
        print(f"Fitting {len(cell_ids)} cells...")
        bounding_box = _merge_bounding_boxes(bounding_boxes, cell_ids)
        bounding_box.expand(x=blur_radius, y=blur_radius, z=0)

        mask = create_mask_for(Image(image))
        mask.set_bounds(bounding_box)
        if mask.has_zero_volume():
            continue

        gaussians = []
        for cell_id in cell_ids:
            mask.add_from_labeled(watershed_image, cell_id + 1)  # Background is 0, so cell 0 uses color 1

            center_zyx = position_centers[cell_id + 1]
            if numpy.any(numpy.isnan(center_zyx)):
                print("No center of mass for cell " + str(center_zyx))
                continue  # No center of mass for this cell id
            intensity = image[int(center_zyx[0]), int(center_zyx[1]), int(center_zyx[2])]
            gaussians.append(Gaussian(intensity, center_zyx[2], center_zyx[1], center_zyx[0], 50, 50, 2, 0, 0, 0))
        mask.dilate_xy(blur_radius // 2)
        cropped_image = mask.create_masked_image(Image(image))
        smoothing.smooth(cropped_image, blur_radius)

        offset_x, offset_y, offset_z = bounding_box.min_x, bounding_box.min_y, bounding_box.min_z
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
    return all_gaussians


def _merge_bounding_boxes(all_boxes: ndarray, cell_ids: List[int]) -> BoundingBox:
    """Creates a bounding box object that encompasses the bounding boxes of all the given cells."""
    combined_bounding_box = None
    for cell_id in cell_ids:
        bounding_box = all_boxes[cell_id + 1]  # Background is 0, so cell 0 uses color 1
        if combined_bounding_box is None:
            combined_bounding_box = bounding_box
            continue
        combined_bounding_box[0] = min(combined_bounding_box[0], bounding_box[0])
        combined_bounding_box[1] = max(combined_bounding_box[1], bounding_box[1])
        combined_bounding_box[2] = min(combined_bounding_box[2], bounding_box[2])
        combined_bounding_box[3] = max(combined_bounding_box[3], bounding_box[3])
        combined_bounding_box[4] = min(combined_bounding_box[4], bounding_box[4])
        combined_bounding_box[5] = max(combined_bounding_box[5], bounding_box[5])
    return bounding_box_from_mahotas(combined_bounding_box)


def _dilate(image_3d: ndarray):
    scratch_2d = numpy.empty_like(image_3d[0])
    kernel = numpy.ones((5, 5), numpy.uint8)
    for z in range(image_3d.shape[0]):
        cv2.dilate(image_3d[z], kernel, dst=scratch_2d, iterations=2)
        image_3d[z] = scratch_2d


