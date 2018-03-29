from typing import Set

from imaging import Particle


class Parameters:
    """Object used to store measurement parameters."""

    shape_detection_radius: int  # Used for detecting shapes
    intensity_detection_radius: int  # Used for detection (changed) intensities.
    max_distance: int  # Maximum distance between mother and daughter cells
    intensity_detection_radius_large: int

    def __init__(self, **kwargs):
        """Sets all given keyword args as parameters on this object."""
        for name, value in kwargs.items():
            setattr(self, name, value)


class Family:
    """A mother cell with two daughter cells."""
    mother: Particle
    daughters: Set[Particle]

    def __init__(self, mother: Particle, daughter1: Particle, daughter2: Particle):
        self.mother = mother
        self.daughters = {daughter1, daughter2}

    @staticmethod
    def _pos_str(particle: Particle) -> str:
        return "(" + ("%.2f" % particle.x) + ", " + ("%.2f" % particle.y) + ", " + ("%.0f" % particle.z) + ")"

    def __str__(self):
        return self._pos_str(self.mother) + " " + str(self.mother.time_point_number()) + "---> " \
               + " and ".join([self._pos_str(daughter) for daughter in self.daughters])

    def __repr__(self):
        return "Family(" + repr(self.mother) + ", " +  ", ".join([repr(daughter) for daughter in self.daughters]) + ")"

    def __hash__(self):
        hash_code = hash(self.mother)
        for daughter in self.daughters:
            hash_code += hash(daughter)
        return hash_code

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and other.mother == self.mother \
            and other.daughters == self.daughters