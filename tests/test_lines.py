from unittest import TestCase

from ai_track.core.vector import Vector3
from ai_track.imaging import lines, angles
from ai_track.imaging.lines import Line3


class TestLines(TestCase):

    def test_direction_along_z_line(self):
        line_in_z_direction = Line3(Vector3(2, 2, 2), Vector3(2, 2, 6))

        self.assertEqual(270, lines.direction_to_point(line_in_z_direction, Vector3(0, 2, 2)))
        self.assertEqual(90, lines.direction_to_point(line_in_z_direction, Vector3(8, 2, 2)))

        # Make sure this matches the 2D case
        self.assertEqual(270, angles.direction_2d(line_in_z_direction.point, Vector3(0, 2, 2)))
        self.assertEqual(90, angles.direction_2d(line_in_z_direction.point, Vector3(8, 2, 2)))

    def test_direction_along_x_line(self):
        line_along_x_axis = Line3(Vector3(5, 0, 0), Vector3(6, 0, 0.001))

        self.assertEqual(0, lines.direction_to_point(line_along_x_axis, Vector3(2, -2, 0)))
        self.assertEqual(45, lines.direction_to_point(line_along_x_axis, Vector3(2, -2, -2)))
        self.assertEqual(90, lines.direction_to_point(line_along_x_axis, Vector3(2, 0, -2)))
        self.assertEqual(180, lines.direction_to_point(line_along_x_axis, Vector3(2, 2, 0)))
        self.assertEqual(270, lines.direction_to_point(line_along_x_axis, Vector3(2, 0, 2)))
