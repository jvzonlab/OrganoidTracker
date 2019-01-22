"""Some helper function for calculations with angles."""
import math

from autotrack.core.position import Position


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
