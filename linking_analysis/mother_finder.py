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