"""Some helper function for calculations with angles."""
import math

from autotrack.core.particles import Particle


def direction_2d(particle1: Particle, particle2: Particle) -> float:
    """Gets the direction from the given particle to the given particle, with 0* pointing upwards."""
    return (90 + math.degrees(math.atan2(particle2.y - particle1.y, particle2.x - particle1.x))) % 360


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
