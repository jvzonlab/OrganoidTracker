from typing import Optional, List, Iterable

from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.score import Score, ScoreCollection, Family
from autotrack.linking import cell_cycle
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.errors import Error


def apply(scores: ScoreCollection, particles: ParticleCollection, graph: Graph):
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters."""
    for particle in graph:
        error = get_error(graph, particle, scores, particles)
        linking_markers.set_error_marker(graph, particle, error)


def get_error(links: Graph, particle: Particle, scores: ScoreCollection, particles: ParticleCollection) -> Optional[Error]:
    future_particles = _get_future_particles(links, particle)
    if len(future_particles) > 2:
        return Error.TOO_MANY_DAUGHTER_CELLS
    elif len(future_particles) == 0 \
            and particle.time_point_number() < particles.last_time_point_number() \
            and linking_markers.get_track_end_marker(links, particle) is None:
        return Error.NO_FUTURE_POSITION
    elif len(future_particles) == 2:
        if scores.has_scores():
            score = scores.of_family(Family(particle, *future_particles))
            if score is None or score.is_unlikely_mother():
                return Error.LOW_MOTHER_SCORE
        age = cell_cycle.get_age(links, particle)
        if age is not None and age < 5:
            return Error.YOUNG_MOTHER

    past_particles = _get_past_particles(links, particle)
    if len(past_particles) == 0:
        if particle.time_point_number() > particles.first_time_point_number() \
                and linking_markers.get_track_start_marker(links, particle) is None:
            return Error.NO_PAST_POSITION
    elif len(past_particles) >= 2:
        return Error.CELL_MERGE
    else:  # len(past_particles) == 1
        # Check cell size
        past_particle = past_particles[0]
        future_particles_of_past_particle = _get_future_particles(links, past_particle)
        shape = particles.get_shape(particle)
        if not shape.is_unknown() and len(future_particles_of_past_particle) == 1:
            past_shape = particles.get_shape(past_particle)
            if not past_shape.is_unknown() and past_shape.volume() / (shape.volume() + 0.0001) > 3:
                return Error.SHRUNK_A_LOT
    return None


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


def _set_error(graph: Graph, particle: Particle, error: Error):
    graph.add_node(particle, error=error.value)


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


def apply_on(experiment: Experiment, *iterable: Particle):
    """Adds errors for all logical inconsistencies for particles in the collection, like cells that spawn out of
    nowhere, cells that merge together and cells that have three or more daughters."""
    graph = experiment.links.graph
    for particle in iterable:
        error = get_error(graph, particle, experiment.scores, experiment.particles)
        linking_markers.set_error_marker(graph, particle, error)
