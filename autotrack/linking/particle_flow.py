from typing import Tuple

import numpy
from networkx import Graph

from autotrack.core import TimePoint, Particle
from autotrack.linking.link_fixer import find_preferred_past_particle, find_preferred_future_particles


def get_flow_to_previous(graph: Graph, time_point: TimePoint, center: Particle,
                         max_dx_and_dy: int = 50, max_dz = 2) -> Tuple[float, float, float]:
    """Gets the average flow of the particles within the specified radius towards the previous time point. Returns
    (0,0,0) if there are no particles. The given center particle must be in the givne time point.
    """

    count = 0
    total_movement = numpy.zeros(3)
    for particle in time_point.particles():
        if _is_far_way_or_same(center, particle, max_dx_and_dy, max_dz):
            continue

        past_position = find_preferred_past_particle(graph, particle)
        if past_position is None:
            continue

        total_movement += (past_position.x - particle.x, past_position.y - particle.y, past_position.z - particle.z)
        count += 1

    if count == 0:
        return (0, 0, 0)
    return total_movement[0] / count, total_movement[1] / count, total_movement[2] / count


def get_flow_to_next(graph: Graph, time_point: TimePoint, center: Particle,
                         max_dx_and_dy: int = 50, max_dz = 2) -> Tuple[float, float, float]:
    """Gets the average flow of the particles within the specified radius towards the next time point. Returns
    (0,0,0) if there are no particles. The given center particle must be in the givne time point. Ignores cell
    divisions and dead cells.
    """

    count = 0
    total_movement = numpy.zeros(3)
    for particle in time_point.particles():
        if _is_far_way_or_same(center, particle, max_dx_and_dy, max_dz):
            continue

        next_positions = find_preferred_future_particles(graph, particle)
        if len(next_positions) != 1:
            continue  # Cell division or dead cell; ignore
        next_position = next_positions.pop()

        total_movement += (next_position.x - particle.x, next_position.y - particle.y, next_position.z - particle.z)
        count += 1

    if count == 0:
        return (0, 0, 0)
    return total_movement[0] / count, total_movement[1] / count, total_movement[2] / count


def _is_far_way_or_same(center: Particle, particle: Particle, max_dx_and_dy: int, max_dz: int) -> bool:
    if particle == center:
        return True  # The center is ignored
    if abs(particle.x - center.x) > max_dx_and_dy:
        return True
    if abs(particle.y - center.y) > max_dx_and_dy:
        return True
    if abs(particle.z - center.z) > max_dz:
        return True
    return False
