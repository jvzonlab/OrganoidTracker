from typing import Tuple

import numpy
from networkx import Graph

from imaging import TimePoint, Particle
from linking.link_fixer import find_preferred_past_particle


def get_flow_to_previous(graph: Graph, time_point: TimePoint, center: Particle,
                         max_dx_and_dy: int = 50, max_dz = 2) -> Tuple[float, float, float]:
    """Gets the average flow of the particles within the specified radius towards the previous time point. Returns
    (0,0,0) if there are no particles. The given center particle must be in the givne time point.
    """
    center_x = center.x
    center_y = center.y
    center_z = center.z

    count = 0
    total_movement = numpy.zeros(3)
    for particle in time_point.particles():
        if particle == center:
            continue  # The center is ignored
        if abs(particle.x - center_x) > max_dx_and_dy:
            continue
        if abs(particle.y - center_y) > max_dx_and_dy:
            continue
        if abs(particle.z - center_z) > max_dz:
            continue

        past_position = find_preferred_past_particle(graph, particle)
        if past_position is None:
            continue

        total_movement += (past_position.x - particle.x, past_position.y - particle.y, past_position.z - particle.z)
        count += 1

    if count == 0:
        return (0, 0, 0)
    return total_movement[0] / count, total_movement[1] / count, total_movement[2] / count
