"""Some helper function for calculations with angles."""
from imaging import Particle
import math


def direction_2d(particle1: Particle, particle2: Particle) -> float:
    """Gets the direction from the given particle to the given particle, with 0* pointing upwards."""
    return (90 + math.degrees(math.atan2(particle2.y - particle1.y, particle2.x - particle1.x))) % 360

def difference(angle1: float, angle2: float) -> float:
    """Gets the difference between two angles. The number returned is always between 0 and 180, inclusive."""
    difference = abs(angle1 - angle2)
    if difference > 180:
        difference = 360 - difference
    return difference
