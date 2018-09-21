import unittest
from typing import Tuple

import math
import numpy
from numpy import ndarray

from autotrack.segmentation.iso_intensity_curvature import get_negative_gaussian_curvatures, ImageDerivatives

SQRT_OF_2PI = math.sqrt(2 * math.pi)


def _add_3d_gaussian(array: ndarray, intensity: float, mean: Tuple[float, float, float], sd: Tuple[float, float, float]):
    """A three-dimensional Gaussian. Cannot be rotated. Integrating it gives the value of intensity."""
    shape = array.shape
    for z in range(0, shape[0]):
        for y in range(0, shape[1]):
            for x in range(0, shape[2]):
                array[z, y, x] += intensity * _1d_gaussian(x, mean[0], sd[0]) *\
                                  _1d_gaussian(y, mean[1], sd[1]) * _1d_gaussian(z, mean[2], sd[2])


def _1d_gaussian(x, mean, sd):
    return 1 / (sd * SQRT_OF_2PI) * math.exp(-0.5 * ((x - mean) / sd) ** 2)


class TestIsoIntensityCurvature(unittest.TestCase):

    def test(self):
        array = numpy.zeros((20, 25, 25), dtype=numpy.uint8)
        _add_3d_gaussian(array, intensity=90000, mean=(8, 10, 5), sd=(3, 4, 2))
        _add_3d_gaussian(array, intensity=80000, mean=(15, 11, 7), sd=(3, 4, 2))
        out = numpy.full_like(array, 255)
        get_negative_gaussian_curvatures(array, ImageDerivatives(), out, blur_radius=3)

        # Check a few pixel intensities (remember, order is z, y, x)
        self.assertEquals(255, out[7, 12, 6])
        self.assertEquals(0, out[7, 11, 11])
        self.assertEquals(255, out[7, 12, 18])
        # If tests fail, use tifffile.imsave to inspect

