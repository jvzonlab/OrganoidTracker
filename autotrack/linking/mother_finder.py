import itertools
from typing import Set, List

from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.image_loader import ImageLoader
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.score import Family, ScoreCollection
from autotrack.linking.scoring_system import MotherScoringSystem


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
        if len(future_particles) < 2:
            continue
        if warn_on_many_daughters:
            # Only two daughters are allowed
            families.append(Family(particle, future_particles.pop(), future_particles.pop()))
            if len(future_particles) > 2:
                print("Illegal mother: " + str(len(future_particles)) + " daughters found")
        else:
            # Many daughters are allowed
            for daughter1, daughter2 in itertools.combinations(future_particles, 2):
                families.append(Family(particle, daughter1, daughter2))

    return families


def calculates_scores(image_loader: ImageLoader, particles: ParticleCollection, graph: Graph,
                      scoring_system: MotherScoringSystem) -> ScoreCollection:
    """Finds all families in the given links and calculates their scores."""
    scores = ScoreCollection()
    families = find_families(graph, warn_on_many_daughters=False)
    i = 0
    for family in families:
        scores.set_family_score(family, scoring_system.calculate(image_loader, particles, family))
        i += 1
        if i % 100 == 0:
            print("   working on " + str(i) + "/" + str(len(families)) + "...")
    return scores
