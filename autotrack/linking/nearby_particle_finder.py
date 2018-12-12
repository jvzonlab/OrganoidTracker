"""Contains function that allows you to find the nearest few particles"""

import operator
from typing import Iterable, List, Optional, Set

from autotrack.core import TimePoint
from autotrack.core.particles import ParticleCollection, Particle


class _NearestParticles:
    """Internal class for bookkeeping of what the nearest few particles are"""

    def __init__(self, tolerance: float):
        self._tolerance_squared = tolerance ** 2
        self._nearest = {}
        self._shorted_distance_squared = float("inf")

    def add_candidate(self, new_particle: Particle, distance_squared: float) -> None:
        if distance_squared > self._tolerance_squared * self._shorted_distance_squared:
            return # Particle is too far away compared to nearest

        if distance_squared < self._shorted_distance_squared:
            # New shortest distance, remove particles that do not conform
            self._shorted_distance_squared = distance_squared
            self._prune()

        self._nearest[new_particle] = distance_squared

    def _prune(self):
        """Removes all cells with distances greater than shortest_distance * tolerance. Useful when the shortest
        distance just changed."""
        max_allowed_distance_squared = self._shorted_distance_squared * self._tolerance_squared
        for particle in list(self._nearest.keys()): # Iterating over copy of keys to avoid a RuntimeError
            its_distance_squared = self._nearest[particle]
            if its_distance_squared > max_allowed_distance_squared:
                del self._nearest[particle]

    def get_particles(self, max_amount: int) -> List[Particle]:
        """Gets the found particles."""
        items = sorted(self._nearest.items(), key=operator.itemgetter(1))
        particles = [item[0] for item in items]
        if len(particles) > max_amount:
            return particles[0:max_amount]
        return particles


def find_close_particles(particles: Iterable[Particle], around: Particle, tolerance: float,
                         max_amount: int = 1000) -> List[Particle]:
    """Finds the particles nearest to the given particle.

    - search_in is the time_point to search in
    - around is the particle to search around
    - tolerance is a number that influences how much particles other than the nearest are included. A tolerance of 1.05
      makes particles that are 5% further than the nearest also included.
    - max_amount if the maximum amount of returned particles. If there are more particles within the tolerance, then
      only the nearest particles are returned.

    Returns a list of the nearest particles, ordered from closest to furthest
    """
    if tolerance < 1:
        raise ValueError()
    nearest_particles = _NearestParticles(tolerance)
    for particle in particles:
        nearest_particles.add_candidate(particle, particle.distance_squared(around, z_factor=3))
    return nearest_particles.get_particles(max_amount)


def find_closest_particle(particles: Iterable[Particle], around: Particle, ignore_z: bool = False,
                          max_distance: int = 100000) -> Optional[Particle]:
    """Gets the particle closest ot the given position."""
    closest_particle = None
    closest_distance_squared = max_distance ** 2

    for particle in particles:
        if ignore_z:
            around.z = particle.z  # Make search ignore z
        distance = particle.distance_squared(around)
        if distance < closest_distance_squared:
            closest_distance_squared = distance
            closest_particle = particle

    return closest_particle


def find_closest_n_particles(particles: Iterable[Particle], around: Particle, max_amount: int,
                             max_distance: int = 100000) -> Set[Particle]:
    max_distance_squared = max_distance ** 2
    closest_particles = []

    for particle in particles:
        distance_squared = particle.distance_squared(around)
        if distance_squared > max_distance_squared:
            continue
        if len(closest_particles) < max_amount or closest_particles[-1][0] > distance_squared:
            # Found closer particle
            closest_particles.append((distance_squared, particle))
            closest_particles.sort(key=operator.itemgetter(0))
            if len(closest_particles) > max_amount:
                # List too long, remove furthest
                del closest_particles[-1]

    return_value = set()
    for distance_squared, particle in closest_particles:
        return_value.add(particle)
    return return_value
