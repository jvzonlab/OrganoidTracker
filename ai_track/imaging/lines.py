"""Oper"""

from ai_track.core.vector import Vector3


class Line3:
    """Defines a line in a three dimensional space."""
    point: Vector3  # Starting point of the line
    direction: Vector3  # Direction vector, not necessarily normalized

    def __init__(self, point1: Vector3, point2: Vector3):
        self.point = point1
        self.direction = point2 - point1

    @staticmethod
    def from_points(point1: Vector3, point2: Vector3):
        """Defines a line using two points."""
        return Line3(point1, point2)

    def __repr__(self) -> str:
        return "Line3(point1=" + repr(self.point) + ", point2=" + repr(self.point + self.direction) + ")"


def point_on_line_2_nearest_to_line_1(*, line_1: Line3, line_2: Line3):
    """Finds the point on line 2 that is the nearest point towards line 1. See Wikipedia on skew lines for the formula.
    """
    try:
        n1 = line_1.direction.cross(line_2.direction.cross(line_1.direction))
        return line_2.point + line_2.direction.multiply(
            ((line_1.point - line_2.point).dot(n1))
                         /
            (line_2.direction.dot(n1)))
    except ZeroDivisionError:
        # Lines are parallel, return random point
        return line_2.point
