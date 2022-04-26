import unittest

import numpy

from organoid_tracker.core.position import Position
from organoid_tracker.core.spline import Spline


class TestDataAxis(unittest.TestCase):

    def test_horizontal_path(self):
        path = Spline()
        path.add_point(0, 0, 0)
        path.add_point(10, 0, 0)
        path.add_point(20, 0, 0)

        self.assertEqual(0, path.get_z())
        self.assertEqual(">", path.get_direction_marker())
        self.assertEqual([0, 10, 20], path.get_points_2d()[0])  # Checks if x coords match
        self.assertEqual(15, path.to_position_on_axis(Position(15, 0, 0)).pos)  # Test a position on the path
        self.assertEqual(19, path.to_position_on_axis(Position(19, 3, 3)).pos)  # Test a position next to the path

    def test_vertical_path(self):
        path = Spline()
        path.add_point(0, 0, 0)
        path.add_point(0, 10, 0)
        path.add_point(0, 20, 0)

        self.assertEqual("v", path.get_direction_marker())

    def test_diagonal_path(self):
        path = Spline()
        path.add_point(0, 0, 0)
        path.add_point(1, 1, 0)
        path.add_point(2, 2, 0)

        self.assertEqual(numpy.sqrt(2), path.to_position_on_axis(Position(1, 1, 0)).pos)

    def test_segments_of_different_length(self):
        path = Spline()
        path.add_point(0, 0, 0)
        path.add_point(10, 0, 0)
        path.add_point(30, 20, 0)
        path.add_point(32, 20, 0)

        # Get a point on the curve
        self.assertAlmostEqual(10.44, path.to_position_on_axis(Position(10, 0, 0)).pos, places=2)
        x, y = path.from_position_on_axis(10.44)
        self.assertAlmostEqual(10, x, places=1)
        self.assertAlmostEqual(0, y, places=1)

    def test_reposition_offset(self):
        path = Spline()
        path.add_point(0, 0, 0)
        path.add_point(10, 0, 0)
        path.add_point(20, 0, 0)

        self.assertEqual(5, path.to_position_on_axis(Position(5, 0, 0)).pos)
        path.update_offset_for_positions([Position(5, 0, 0), Position(6, 0, 0)])
        self.assertEqual(0, path.to_position_on_axis(Position(5, 0, 0)).pos)  # Make sure zero-point has moved
