"""Some helper function for calculations with angles."""
import math

from ai_track.core.position import Position
from ai_track.core.resolution import ImageResolution
from ai_track.core.vector import Vector3

_ZERO_POS = Position(0, 0, 0)

def direction_2d(position1: Position, position2: Position) -> float:
    """Gets the direction from the given position to the given position, with 0* pointing upwards."""
    return (90 + math.degrees(math.atan2(position2.y - position1.y, position2.x - position1.x))) % 360


def difference(angle1: float, angle2: float) -> float:
    """Gets the difference between two angles. The number returned is always between 0 inclusive and 180 exclusive."""
    difference = abs(angle1 - angle2)
    if difference >= 180:
        difference = 360 - difference
    return difference


def direction_change(angle1: float, angle2: float) -> float:
    """Gets how many degrees you need to turn to reach angle2, starting from angle1. The result will be from -180 to
    +180."""
    difference = angle2 - angle1
    if difference > 180:
        difference = -(360 - difference)
    elif difference < -180:
        difference+= 360
    return difference


def direction_change_of_line(angle1: float, angle2: float) -> float:
    """Returns a value from 0 to 90 to how much the direction needs to change to go from angle1 to angle2. Flipped
    angles are considered equal, so 10 degrees is equal to 190 degrees. To go from 5 degrees to 190 degrees, you
    therefore need to turn only 5 degrees."""
    # 170 and 190: 20 degrees to the right
    # 350 and 10: 20 degrees to the right

    change = abs(direction_change(angle1, angle2))
    change_flipped = abs(direction_change(angle1, flipped(angle2)))
    return min(change, change_flipped)


def flipped(angle: float) -> float:
    """Gets the direction flipped 180*, so exactly in the opposite direction. The returned direction is from 0
    (inclusive) to 360 (exclusive).
    """
    return (angle + 180) % 360


def mirrored(angle: float, mirror_angle: float) -> float:
    """Mirrors the angle against the given plane. See the example for a 0* mirror plane:

        Old     New
        ^   |   ^
         \  |  /
          \ | /
           \|/

    A 0* mirror is equivalent to a 180* mirror. (Same for any other angle.) The returned direction is from 0 (inclusive)
    to 360 (exclusive).
    """
    change = direction_change(angle, mirror_angle)
    if change > 90:
        mirror_angle = flipped(mirror_angle)
        change = direction_change(angle, mirror_angle)
    return (mirror_angle + change) % 360


def right_hand_rule(a: Vector3, b: Vector3, c: Vector3) -> float:
    """Returns the angle formed by A -> B -> C using the 'right-hand rule' from B. Based on
    https://math.stackexchange.com/questions/361412/finding-the-angle-between-three-points . The result is
    equal to angle_between_vectors(b - a, c - b), except that 180 is returned if a, b and c lie on a straight line."""
    ab_dot_bc = (b - a).dot(c - b)

    length_ab = a.distance(b)
    length_bc = b.distance(c)

    try:
        cos_value = ab_dot_bc / (length_ab * length_bc)
    except ZeroDivisionError as e:
        # Better error message
        raise ValueError(f"Error calculating angle for {a} {b} {c}: {e}")

    # Correct for rounding errors in float calculations causing a math domain error
    if 1 <= cos_value <= 1.0000000000000004:
        return 0
    elif -1 >= cos_value >= -1.0000000000000004:
        return 0

    try:
        return math.degrees(math.acos(cos_value))
    except ValueError as e:
        # Better error message
        raise ValueError(f"Error calculating angle for {a} {b} {c}: {e}")


def angle_between_vectors(vector1: Vector3, vector2: Vector3) -> float:
    """Returns the angle between the two vectors. Will be between 0 and 180."""
    length1 = vector1.length()
    length2 = vector2.length()

    return math.degrees(math.acos((vector1.dot(vector2) / (length1 * length2))))
