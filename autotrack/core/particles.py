from operator import itemgetter
from typing import Dict, AbstractSet, Optional, Iterable, Set

import math

from autotrack.core import TimePoint
from autotrack.core.resolution import ImageResolution
from autotrack.core.shape import ParticleShape, UnknownShape


class Particle:
    """A detected particle. Only the 3D + time position is stored here, see the ParticleShape class for the shape."""
    x: float
    y: float
    z: float
    _time_point_number: Optional[int] = None

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance_squared(self, other: "Particle", z_factor: float = 5) -> float:
        """Gets the squared distance. Working with squared distances instead of normal ones gives a much better
        performance, as the expensive sqrt(..) function can be avoided."""
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + ((self.z - other.z) * z_factor) ** 2

    def distance_um(self, other: "Particle", resolution: ImageResolution) -> float:
        """Gets the distance to the other particle in micrometers."""
        dx = (self.x - other.x) * resolution.pixel_size_zyx_um[2]
        dy = (self.y - other.y) * resolution.pixel_size_zyx_um[1]
        dz = (self.z - other.z) * resolution.pixel_size_zyx_um[0]
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def time_point_number(self) -> Optional[int]:
        return self._time_point_number

    def with_time_point_number(self, time_point_number: int) -> "Particle":
        if self._time_point_number is not None and self._time_point_number != time_point_number:
            raise ValueError("time_point_number was already set")
        self._time_point_number = int(time_point_number)
        return self

    def with_time_point(self, time_point: TimePoint) -> "Particle":
        return self.with_time_point_number(time_point.time_point_number())

    def __repr__(self):
        string = "Particle(" + ("%.2f" % self.x) + ", " + ("%.2f" % self.y) + ", " + ("%.0f" % self.z) + ")"
        if self._time_point_number is not None:
            string += ".with_time_point_number(" + str(self._time_point_number) + ")"
        return string

    def __str__(self):
        string = "cell at (" + ("%.2f" % self.x) + ", " + ("%.2f" % self.y) + ", " + ("%.2f" % self.z) + ")"
        if self._time_point_number is not None:
            string += " at time point " + str(self._time_point_number)
        return string

    def __hash__(self):
        if self._time_point_number is None:
            return hash(int(self.x)) ^ hash(int(self.y)) ^ hash(int(self.z))
        return hash(int(self.x)) ^ hash(int(self.y)) ^ hash(int(self.z)) ^ hash(int(self._time_point_number))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and abs(self.x - other.x) < 0.00001 and abs(self.x - other.x) < 0.00001 and abs(self.z - other.z) < 0.00001 \
               and self._time_point_number == other._time_point_number

    def time_point(self):
        """Gets the time point of this particle."""
        return TimePoint(self._time_point_number)


class _ParticlesAtTimePoint:
    """Holds the particles of a single point in time."""

    _particles: Dict[Particle, ParticleShape]

    def __init__(self):
        self._particles = dict()
        self._mother_scores = dict()

    def particles(self) -> AbstractSet[Particle]:
        return self._particles.keys()

    def particles_and_shapes(self) -> Dict[Particle, ParticleShape]:
        return self._particles

    def get_shape(self, particle: Particle) -> ParticleShape:
        """Gets the shape of a particle. Returns UnknownShape if the given particle is not part of this time point."""
        shape = self._particles.get(particle)
        if shape is None:
            return UnknownShape()
        return shape

    def add_particle(self, particle: Particle, particle_shape: Optional[ParticleShape]):
        """Adds a particle to this time point. If the particle was already added, but a shape was provided, its shape is
        replaced."""
        if particle_shape is None:
            if particle in self._particles:
                return  # Don't overwrite known shape with an unknown shape.
            particle_shape = UnknownShape()  # Don't use None as value in the dict
        self._particles[particle] = particle_shape

    def detach_particle(self, particle: Particle):
        """Removes a single particle. Raises KeyError if that particle was not in this time point. Does not remove a
        particle from the linking graph. See also Experiment.remove_particle."""
        del self._particles[particle]

    def is_empty(self):
        """Returns True if there are no particles stored."""
        return len(self._particles) == 0


class ParticleCollection:

    _all_particles: Dict[int, _ParticlesAtTimePoint]
    _min_time_point_number: Optional[int] = None
    _max_time_point_number: Optional[int] = None

    def __init__(self):
        self._all_particles = dict()

    def of_time_point(self, time_point: TimePoint) -> AbstractSet[Particle]:
        """Returns all particles for a given time point. Returns an empty set if that time point doesn't exist."""
        particles_at_time_point = self._all_particles.get(time_point.time_point_number())
        if not particles_at_time_point:
            return set()
        return particles_at_time_point.particles()

    def detach_all_for_time_point(self, time_point: TimePoint):
        """Removes all particles for a given time point, if any."""
        if time_point.time_point_number() in self._all_particles:
            del self._all_particles[time_point.time_point_number()]
            self._update_min_max_time_points_for_removal()

    def add(self, particle: Particle, shape: Optional[ParticleShape] = None):
        """Adds a particle, optionally with the given shape. The particle must have a time point specified."""
        time_point_number = particle.time_point_number()
        if time_point_number is None:
            raise ValueError("Particle does not have a time point, so it cannot be added")

        self._update_min_max_time_points_for_addition(time_point_number)

        particles_at_time_point = self._all_particles.get(time_point_number)
        if particles_at_time_point is None:
            particles_at_time_point = _ParticlesAtTimePoint()
            self._all_particles[time_point_number] = particles_at_time_point
        particles_at_time_point.add_particle(particle, shape)

    def _update_min_max_time_points_for_addition(self, new_time_point_number: int):
        """Bookkeeping: makes sure the min and max time points are updated when a new time point is added"""
        if self._min_time_point_number is None or new_time_point_number < self._min_time_point_number:
            self._min_time_point_number = new_time_point_number
        if self._max_time_point_number is None or new_time_point_number > self._max_time_point_number:
            self._max_time_point_number = new_time_point_number

    def _update_min_max_time_points_for_removal(self):
        """Bookkeeping: recalculates min and max time point if a time point was removed."""
        # Reset min and max, then repopulate by readding all time points
        self._min_time_point_number = None
        self._max_time_point_number = None
        for time_point_number in self._all_particles.keys():
            self._update_min_max_time_points_for_addition(time_point_number)

    def detach_particle(self, particle: Particle):
        """Removes a particle from a time point."""
        particles_at_time_point = self._all_particles.get(particle.time_point_number())
        if particles_at_time_point is None:
            return

        particles_at_time_point.detach_particle(particle)

        # Remove time point entirely
        if particles_at_time_point.is_empty():
            del self._all_particles[particle.time_point_number()]
            self._update_min_max_time_points_for_removal()

    def of_time_point_with_shapes(self, time_point: TimePoint) -> Dict[Particle, ParticleShape]:
        """Gets all particles and shapes of a time point. New particles must be added using self.add(...), not using
        this dict."""
        particles_at_time_point = self._all_particles.get(time_point.time_point_number())
        if not particles_at_time_point:
            return dict()
        return particles_at_time_point.particles_and_shapes()

    def get_shape(self, particle: Particle) -> ParticleShape:
        particles_at_time_point = self._all_particles.get(particle.time_point_number())
        if particles_at_time_point is None:
            return UnknownShape()
        return particles_at_time_point.get_shape(particle)

    def first_time_point_number(self) -> Optional[int]:
        """Gets the first time point that contains particles, or None if there are no particles stored."""
        return self._min_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        """Gets the last time point (inclusive) that contains particles, or None if there are no particles stored."""
        return self._max_time_point_number

    def exists(self, particle: Particle) -> bool:
        """Returns whether the given particle is part of the experiment."""
        particles_at_time_point = self._all_particles.get(particle.time_point_number())
        if particles_at_time_point is None:
            return False
        return particle in particles_at_time_point.particles()


def get_closest_particle(particles: Iterable[Particle], search_position: Particle,
                         ignore_z: bool = False, max_distance: int = 100000) -> Optional[Particle]:
    """Gets the particle closest ot the given position."""
    closest_particle = None
    closest_distance_squared = max_distance ** 2

    for particle in particles:
        if ignore_z:
            search_position.z = particle.z  # Make search ignore z
        distance = particle.distance_squared(search_position)
        if distance < closest_distance_squared:
            closest_distance_squared = distance
            closest_particle = particle

    return closest_particle


def get_closest_n_particles(particles: Iterable[Particle], search_position: Particle, amount: int,
                            max_distance: int = 100000) -> Set[Particle]:
    max_distance_squared = max_distance ** 2
    closest_particles = []

    for particle in particles:
        distance_squared = particle.distance_squared(search_position)
        if distance_squared > max_distance_squared:
            continue
        if len(closest_particles) < amount or closest_particles[-1][0] > distance_squared:
            # Found closer particle
            closest_particles.append((distance_squared, particle))
            closest_particles.sort(key=itemgetter(0))
            if len(closest_particles) > amount:
                # List too long, remove furthest
                del closest_particles[-1]

    return_value = set()
    for distance_squared, particle in closest_particles:
        return_value.add(particle)
    return return_value
