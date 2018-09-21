"""Code for fitting cells to Gaussian functions."""
from typing import List, Iterable, Dict, Optional
from timeit import default_timer

import cv2
import numpy
import scipy.optimize
from numpy import ndarray

from autotrack.particle_detection import smoothing, ellipse_cluster

from autotrack.core.gaussian import Gaussian

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


def perform_gaussian_mixture_fit(original_image: ndarray, guesses: Iterable[Gaussian]) -> List[Gaussian]:
    """Fits multiple Gaussians to the image (a Gaussian Mixture Model). Initial seeds must be given."""
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


def perform_gaussian_mixture_fit_from_watershed(image: ndarray, watershed_image: ndarray, blur_radius: int
                                                ) -> List[Gaussian]:
    """GMM using watershed as seeds. The watershed is used to fit as few Gaussians at the same time as possible."""
    ellipse_stacks = ellipse_cluster.get_ellipse_stacks_from_watershed(watershed_image)
    ellipse_clusters = ellipse_cluster.find_overlapping_stacks(ellipse_stacks)

    all_gaussians: List[Optional[Gaussian]] = [None] * len(ellipse_stacks)  # Initialize empty list
    start_time = default_timer()
    for cluster in ellipse_clusters:
        offset_x, offset_y, offset_z, cropped_image = cluster.get_image_for_fit(image, blur_radius)
        if cropped_image is None:
            continue
        smoothing.smooth(cropped_image, blur_radius)
        gaussians = cluster.guess_gaussians(image)
        tags = cluster.get_tags()

        gaussians = [gaussian.translated(-offset_x, -offset_y, -offset_z) for gaussian in gaussians]
        try:
            gaussians = perform_gaussian_mixture_fit(cropped_image, gaussians)
            for i, gaussian in enumerate(gaussians):
                all_gaussians[tags[i]] = gaussian.translated(offset_x, offset_y, offset_z)
        except ValueError:
            print("Minimization failed for " + str(cluster))
            continue
    end_time = default_timer()
    print("Whole fitting process took " + str(end_time - start_time) + " seconds.")
    return all_gaussians


def _dilate(image_3d: ndarray):
    scratch_2d = numpy.empty_like(image_3d[0])
    kernel = numpy.ones((5, 5), numpy.uint8)
    for z in range(image_3d.shape[0]):
        cv2.dilate(image_3d[z], kernel, dst=scratch_2d, iterations=2)
        image_3d[z] = scratch_2d


