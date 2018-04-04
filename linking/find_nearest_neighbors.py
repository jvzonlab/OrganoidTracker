"""Contains function that allows you to find the nearest few particles"""

from imaging import Particle, TimePoint
import operator


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

    def get_particles(self):
        """Gets the found particles."""
        items = sorted(self._nearest.items(), key=operator.itemgetter(1))
        return [item[0] for item in items]


def find_nearest_particles(search_in: TimePoint, around: Particle, tolerance: float):
    """Finds the particles nearest to the given particle.

    - search_in is the time_point to search in
    - around is the particle to search around
    - tolerance is a number that influences how much particles other than the nearest are included. A tolerance of 1.05
      makes particles that are 5% further than the nearest also included.

    Returns a list of the nearest particles, ordered from closest to furthest
    """
    if tolerance < 1:
        raise ValueError()
    nearest_particles = _NearestParticles(tolerance)
    for particle in search_in.particles():
        nearest_particles.add_candidate(particle, particle.distance_squared(around, z_factor=3))
    return nearest_particles.get_particles()

