from networkx import Graph

from imaging import errors, Experiment, Particle


def apply(experiment: Experiment, graph: Graph):
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters."""
    for particle in graph:

        future_particles = _get_future_particles(graph, particle)
        if len(future_particles) > 2:
            _set_error(graph, particle, errors.TOO_MANY_DAUGHTER_CELLS)
        if len(future_particles) == 0 and particle.time_point_number() < experiment.last_time_point_number() - 1:
            _set_error(graph, particle, errors.NO_FUTURE_POSITION)

        past_particles = _get_past_particles(graph, particle)
        if len(past_particles) == 0 and particle.time_point_number() > experiment.first_time_point_number():
            _set_error(graph, particle, errors.NO_PAST_POSITION)
        if len(past_particles) >= 2:
            _set_error(graph, particle, error=errors.CELL_MERGE)


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