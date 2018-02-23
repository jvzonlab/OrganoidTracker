from networkx import Graph
from imaging import Experiment, TimePoint
from linking_analysis.find_nearest_neighbors import find_nearest_particles


def link_particles(experiment: Experiment, tolerance: float = 1.0, min_time_point: int = 0, max_time_point: int = 5000) -> Graph:
    """Simple nearest neighbour linking, keeping a list of potential candidates based on a given tolerance.

    A tolerance of 1.05 also links particles 5% from the closest particle. Note that if a tolerance higher than 1 is
    given, some pruning is needed on the final result.

    max_time_point is the last time point that will still be included.
    """
    graph = Graph()

    time_point_current = experiment.get_time_point(max(experiment.first_time_point_number(), min_time_point))
    _add_nodes(graph, time_point_current)

    try:
        while time_point_current.time_point_number() < max_time_point:
            time_point_previous = time_point_current

            time_point_current = experiment.get_next_time_point(time_point_previous)
            _add_nodes(graph, time_point_current)
            _add_edges(graph, time_point_previous, time_point_current, tolerance)
    except KeyError:
        # Done! No more time points remain
        pass

    print("Done creating nearest-neighbor links!")
    return graph


def _add_nodes(graph: Graph, time_point: TimePoint) -> None:
    for particle in time_point.particles():
        graph.add_node(particle)


def _add_edges(graph: Graph, time_point_previous: TimePoint, time_point_current: TimePoint, tolerance: float):
    for particle in time_point_current.particles():
        nearby_list = find_nearest_particles(time_point_previous, particle, tolerance)
        preferred = True
        for nearby_particle in nearby_list:
            graph.add_edge(particle, nearby_particle, pref=preferred)
            preferred = False # All remaining links are not preferred
