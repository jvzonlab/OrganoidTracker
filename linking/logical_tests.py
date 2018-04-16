from typing import Optional

from networkx import Graph

from imaging import errors, Experiment, Particle, cell, Score


def apply(experiment: Experiment, graph: Graph):
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters."""
    for particle in graph:

        future_particles = _get_future_particles(graph, particle)
        if len(future_particles) > 2:
            _set_error(graph, particle, errors.TOO_MANY_DAUGHTER_CELLS)
        elif len(future_particles) == 0 and particle.time_point_number() < experiment.last_time_point_number() - 1:
            _set_error(graph, particle, errors.NO_FUTURE_POSITION)
        elif len(future_particles) == 2:
            score = _get_highest_mother_score(experiment, particle)
            if score is None or score.is_unlikely_mother():
                _set_error(graph, particle, errors.LOW_MOTHER_SCORE)
            age = cell.get_age(graph, particle)
            if age is not None and age < 2:
                _set_error(graph, particle, errors.YOUNG_MOTHER)

        past_particles = _get_past_particles(graph, particle)
        if len(past_particles) == 0 and particle.time_point_number() > experiment.first_time_point_number():
            _set_error(graph, particle, errors.NO_PAST_POSITION)
        if len(past_particles) >= 2:
            _set_error(graph, particle, errors.CELL_MERGE)


def _get_highest_mother_score(experiment: Experiment, particle: Particle) -> Optional[Score]:
    highest_score = None
    highest_score_num = -999
    for scored_family in experiment.get_time_point(particle.time_point_number()).mother_scores(particle):
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


def _get_past_particles(graph: Graph, particle: Particle):
    try:
        linked_particles = graph[particle]
        return [p for p in linked_particles if p.time_point_number() < particle.time_point_number()]
    except KeyError:
        return []