"""Used to view only part of a linking graph."""

import networkx
from networkx import Graph

from autotrack.core.particles import Particle


def _is_in_time_points(particle: Particle, first_time_point_number: int, last_time_point_number: int) -> bool:
    return particle.time_point_number() >= first_time_point_number and particle.time_point_number() <= last_time_point_number


def limit_to_time_points(graph: Graph, first_time_point_number: int, last_time_point_number: int) -> Graph:
    """Returns a view of the graph consisting of only the nodes between the first and last time point number, inclusive.
    """
    return graph.subgraph([particle for particle in graph.nodes()
                          if _is_in_time_points(particle, first_time_point_number, last_time_point_number)])
