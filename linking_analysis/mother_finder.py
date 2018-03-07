from collections import namedtuple
from typing import Set, Tuple

from networkx import Graph

from imaging import Particle


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


class Family:
    """A mother cell with two daughter cells."""
    mother: Particle
    daughter1: Particle
    daughter2: Particle

    def __init__(self, mother: Particle, daughter1: Particle, daughter2: Particle):
        self.mother = mother
        self.daughter1 = daughter1
        self.daughter2 = daughter2

    @staticmethod
    def _pos_str(particle: Particle) -> str:
        return "(" + ("%.2f" % particle.x) + ", " + ("%.2f" % particle.y) + ", " + ("%.0f" % particle.z) + ")"

    def __str__(self):
        return self._pos_str(self.mother) + " " + str(self.mother.time_point_number()) + "---> " \
               + self._pos_str(self.daughter1) + " and " + self._pos_str(self.daughter2)

    def __repr__(self):
        return "Family(" + repr(self.mother) + ", " + repr(self.daughter1) + ", " + repr(self.daughter2) + ")"

    def __hash__(self):
        return hash(self.mother) ^ hash(self.daughter1) ^ hash(self.daughter2)

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and other.mother == self.mother \
            and other.daughter1 == self.daughter1 \
            and other.daughter2 == self.daughter2


def find_families(graph: Graph) -> Set[Family]:
    """Finds all mother and daughter cells in a graph. Mother cells are cells with at least two daughter cells.
    Returns a set of Family instances.
    """
    families = set()

    for particle in graph.nodes():
        linked_particles = graph[particle]
        future_particles = {p for p in linked_particles if p.time_point_number() > particle.time_point_number()}
        if len(future_particles) >= 2:
            families.add(Family(mother=particle, daughter1=future_particles.pop(), daughter2=future_particles.pop()))
        if len(future_particles) > 2:
            print("Illegal mother: " + str(len(future_particles)) + " daughters found")

    return families