from collections import namedtuple
from typing import Set, Tuple, List

import itertools
from networkx import Graph

from imaging import Particle
from linking import Family


def find_mothers(graph: Graph) -> Set[Particle]:
    """Finds all mother cells in a graph. Mother cells are cells with at least two daughter cells."""
    mothers = set()

    for particle in graph.nodes():
        linked_particles = graph[particle]
        future_particles = {p for p in linked_particles if p.time_point_number() > particle.time_point_number()}
        if len(future_particles) >= 2:
            mothers.add(particle)
        if len(future_particles) > 2:
            print("Illegal mother: " + str(len(future_particles)) + " daughters found")

    return mothers


def find_families(graph: Graph, warn_on_many_daughters = True) -> List[Family]:
    """Finds all mother and daughter cells in a graph. Mother cells are cells with at least two daughter cells.
    Returns a set of Family instances.
    """
    families = list()

    for particle in graph.nodes():
        linked_particles = graph[particle]
        future_particles = {p for p in linked_particles if p.time_point_number() > particle.time_point_number()}
        if len(future_particles) >= 2:
            for daughter1, daughter2 in itertools.combinations(future_particles, 2):
                families.append(Family(particle, daughter1, daughter2))
        if warn_on_many_daughters and len(future_particles) > 2:
            print("Illegal mother: " + str(len(future_particles)) + " daughters found")

    return families