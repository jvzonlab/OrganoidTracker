import unittest

import numpy

from autotrack.core.particles import Particle
from autotrack.core.path import Path


class TestPaths(unittest.TestCase):

    def test_horizontal_path(self):
        path = Path()
        path.add_point(0, 0, 0)
        path.add_point(10, 0, 0)
        path.add_point(20, 0, 0)

        self.assertEquals(0, path.get_z())
        self.assertEquals(">", path.get_direction_marker())
        self.assertEquals([0, 10, 20], path.get_points_2d()[0])  # Checks if x coords match
        self.assertEquals(15, path.get_path_position_2d(Particle(15, 0, 0)).pos)  # Test a position on the path
        self.assertEquals(19, path.get_path_position_2d(Particle(19, 3, 3)).pos)  # Test a position next to the path

    def test_vertical_path(self):
        path = Path()
        path.add_point(0, 0, 0)
        path.add_point(0, 10, 0)
        path.add_point(0, 20, 0)

        self.assertEquals("v", path.get_direction_marker())

    def test_diagonal_path(self):
        path = Path()
        path.add_point(0, 0, 0)
        path.add_point(1, 1, 0)
        path.add_point(2, 2, 0)

        self.assertEquals(numpy.sqrt(2), path.get_path_position_2d(Particle(1, 1, 0)).pos)

    def test_segments_of_different_length(self):
        path = Path()
        path.add_point(0, 0, 0)
        path.add_point(10, 0, 0)
        path.add_point(30, 20, 0)
        path.add_point(32, 20, 0)

        # Get a point on the curve
        self.assertAlmostEqual(10.44, path.get_path_position_2d(Particle(10, 0, 0)).pos, places=2)
        x, y = path.path_position_to_xy(10.44)
        self.assertAlmostEqual(10, x, places=1)
        self.assertAlmostEqual(0, y, places=1)

    def test_reposition_offset(self):
        path = Path()
        path.add_point(0, 0, 0)
        path.add_point(10, 0, 0)
        path.add_point(20, 0, 0)

        self.assertEquals(5, path.get_path_position_2d(Particle(5, 0, 0)).pos)
        path.update_offset_for_particles([Particle(5, 0, 0), Particle(6, 0, 0)])
        self.assertEquals(0, path.get_path_position_2d(Particle(5, 0, 0)).pos)  # Make sure zero-point has moved
