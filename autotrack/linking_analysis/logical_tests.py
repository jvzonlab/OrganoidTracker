from typing import Optional, List

from networkx import Graph

from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.score import Score, ScoreCollection
from autotrack.linking import cell_cycle
from autotrack.linking_analysis import errors


def apply(scores: ScoreCollection, particles: ParticleCollection, graph: Graph):
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters."""
    for particle in graph:

        future_particles = _get_future_particles(graph, particle)
        if len(future_particles) > 2:
            _set_error(graph, particle, errors.TOO_MANY_DAUGHTER_CELLS)
        elif len(future_particles) == 0 and particle.time_point_number() < particles.last_time_point_number() - 1:
            _set_error(graph, particle, errors.NO_FUTURE_POSITION)
        elif len(future_particles) == 2:
            score = _get_highest_mother_score(scores, particle)
            if score is None or score.is_unlikely_mother():
                _set_error(graph, particle, errors.LOW_MOTHER_SCORE)
            age = cell_cycle.get_age(graph, particle)
            if age is not None and age < 5:
                _set_error(graph, particle, errors.YOUNG_MOTHER)

        past_particles = _get_past_particles(graph, particle)
        if len(past_particles) == 0:
            if particle.time_point_number() > particles.first_time_point_number():
                _set_error(graph, particle, errors.NO_PAST_POSITION)
        elif len(past_particles) >= 2:
            _set_error(graph, particle, errors.CELL_MERGE)
        else:  # len(past_particles) == 1
            # Check cell size
            past_particle = past_particles[0]
            shape = particles.get_shape(particle)
            if not shape.is_unknown() and len(future_particles) == 1:
                past_shape = particles.get_shape(past_particle)
                if past_shape.volume() / (shape.volume() + 0.0001) > 3:
                    _set_error(graph, particle, errors.SHRUNK_A_LOT)


def _get_highest_mother_score(scores: ScoreCollection, particle: Particle) -> Optional[Score]:
    highest_score = None
    highest_score_num = -999
    for scored_family in scores.of_time_point(particle.time_point()):
        score = scored_family.score
        score_num = score.total()
        if score_num > highest_score_num:
            highest_score = score
            highest_score_num = score_num
    return highest_score


def _set_error(graph: Graph, particle: Particle, error: int):
    graph.add_node(particle, error=error)
    if "edited" in graph.nodes[particle]:
        # Particle is not edited since addition of error
        del graph.nodes[particle]["edited"]


def _get_future_particles(graph: Graph, particle: Particle):
    try:
        linked_particles = graph[particle]
        return [p for p in linked_particles if p.time_point_number() > particle.time_point_number()]
    except KeyError:
        return []


def _get_past_particles(graph: Graph, particle: Particle) -> List[Particle]:
    try:
        linked_particles = graph[particle]
        return [p for p in linked_particles if p.time_point_number() < particle.time_point_number()]
    except KeyError:
        return []
