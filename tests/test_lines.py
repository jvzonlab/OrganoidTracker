from unittest import TestCase

from organoid_tracker.core.vector import Vector3
from organoid_tracker.imaging import lines, angles
from organoid_tracker.imaging.lines import Line3


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

        self.assertEqual(0, _angle_round(lines.direction_to_point(line_along_x_axis, Vector3(2, -2, 0))))
        self.assertEqual(45, _angle_round(lines.direction_to_point(line_along_x_axis, Vector3(2, -2, -2))))
        self.assertEqual(90, _angle_round(lines.direction_to_point(line_along_x_axis, Vector3(2, 0, -2))))
        self.assertEqual(180, _angle_round(lines.direction_to_point(line_along_x_axis, Vector3(2, 2, 0))))
        self.assertEqual(270, _angle_round(lines.direction_to_point(line_along_x_axis, Vector3(2, 0, 2))))


def _angle_round(angle: float) -> int:
    rounded = int(round(angle))
    if rounded == 360:
        return 0
    return rounded
