import unittest

import numpy

from autotrack.particle_detection.ellipsoid_fit import Ellipsoid


class TestEllipsoidFit(unittest.TestCase):

    def test_inside(self):
        ellipsoid = Ellipsoid(center=numpy.array([3, 4, 5]), radii=numpy.array([2, 3, 2]), rotation=numpy.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]))
        # Test center position using both a tuple and a numpy array
        self.assertTrue(ellipsoid.is_inside((3, 4, 5)))
        self.assertTrue(ellipsoid.is_inside(numpy.array([3, 4, 5])))

        # Some points on the edge
        self.assertTrue(ellipsoid.is_inside(numpy.array([1, 4, 5])))
        self.assertTrue(ellipsoid.is_inside(numpy.array([5, 4, 5])))

        # A few points outside
        self.assertFalse(ellipsoid.is_inside(numpy.array([0.8, 4, 5])))
        self.assertFalse(ellipsoid.is_inside(numpy.array([5.2, 4, 5])))

    def test_inside_futher_away(self):
        ellipsoid = Ellipsoid(center=numpy.array([377.77169329, 181.91924289, 18.23983481]),
                              radii=numpy.array([2.63980712, 10.16912923, 12.24133363]),
                              rotation=numpy.array([
                                  [0.27842013, 0.93987672, -0.19777256],
                                  [0.12658548, 0.16820979, 0.97758968],
                                  [-0.95208106, 0.29721578, 0.07214175]
                              ]))
        self.assertTrue(ellipsoid.is_inside((377, 181, 18)))

