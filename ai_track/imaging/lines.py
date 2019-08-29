"""Oper"""
import math

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

    def translate(self, translation: Vector3) -> "Line3":
        """Gets a translated copy of the line."""
        return Line3(self.point + translation, self.point + translation + self.direction)

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


def direction_to_point(line: Line3, point: Vector3) -> float:
    """Gets the 1D direction of a point towards a line. If the line is pointing upwards (read: parallel with the
    -axis), then this simply returns the same value as angles.direction_2d(vector, point_on_line). Returns a number
    from 0 to 360.

    Note that is doesn't matter in which direction the line is defined.
    """
    # Code based on http://paulbourke.net/geometry/rotate/ (Rotate a point about an arbitrary axis in 3 dimensions)

    # Translate space so that the rotation axis passes through the origin
    q1 = point - line.point
    u = line.direction.normalized()

    d = math.sqrt(u.y * u.y + u.z * u.z)

    # Rotate space about the x axis so that the rotation axis lies in the xz plane
    if d != 0:
        q2 = Vector3(q1.x,
                     q1.y * u.z / d - q1.z * u.y / d,
                     q1.y * u.y / d + q1.z * u.z / d)
    else:
        q2 = q1  # line direction is already aligned to the x axis

    # Rotate space about the y axis so that the rotation axis lies along the z axis
    q1.x = q2.x * d - q2.z * u.x
    q1.y = q2.y
    q1.z = q2.x * u.x + q2.z * d

    # Measure rotation in 2D, now that we have rotated everything
    return (90 + math.degrees(math.atan2(q1.y, q1.x))) % 360


def distance_to_point(line: Line3, search_point: Vector3) -> float:
    """Gets the distnace from the line to the given point. Note that we're calculating the distance to an infinite line,
    which is different from calculating the distance to a line segment."""
    return line.direction.cross(line.point - search_point).length() / line.direction.length()
