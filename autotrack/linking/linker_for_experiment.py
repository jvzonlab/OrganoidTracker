from networkx import Graph

from autotrack import core
from autotrack.core import Experiment, TimePoint, Particle
from autotrack.linking import particle_flow
from autotrack.linking import find_nearest_particles


def nearest_neighbor(experiment: Experiment, tolerance: float = 1.0, min_time_point: int = 0, max_time_point: int = 5000) -> Graph:
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
            _add_nearest_edges(graph, time_point_previous, time_point_current, tolerance)
            _add_nearest_edges_extra(graph, time_point_previous, time_point_current, tolerance)
    except KeyError:
        # Done! No more time points remain
        pass

    print("Done creating nearest-neighbor links!")
    return graph


def nearest_neighbor_using_flow(experiment: Experiment, initial_links: Graph, flow_detection_radius: int,
                                min_time_point: int = 0, max_time_point: int = 5000) -> Graph:
    """A bit more advanced nearest-neighbor linking, that analysis the already-established links around a particle to
    decide in which direction a particle is going.

    max_time_point is the last time point that will still be included.
    """
    graph = _all_links_downgraded(initial_links)

    time_point_current = experiment.get_time_point(max(experiment.first_time_point_number(), min_time_point))

    try:
        while time_point_current.time_point_number() < max_time_point:
            time_point_current = experiment.get_next_time_point(time_point_current)
            _find_nearest_edges_using_flow(graph, initial_links, time_point_current, flow_detection_radius)
    except KeyError:
        # Done! No more time points remain
        pass

    print("Done creating nearest-neighbor links using flow!")
    return graph


def _add_nodes(graph: Graph, time_point: TimePoint) -> None:
    for particle in time_point.particles():
        graph.add_node(particle)


def _add_nearest_edges(graph: Graph, time_point_previous: TimePoint, time_point_current: TimePoint, tolerance: float):
    """Adds edges pointing towards previous time point, making the shortest one the preferred."""
    for particle in time_point_current.particles():
        nearby_list = find_nearest_particles(time_point_previous, particle, tolerance)
        preferred = True
        for nearby_particle in nearby_list:
            graph.add_edge(particle, nearby_particle, pref=preferred)
            preferred = False  # All remaining links are not preferred


def _add_nearest_edges_extra(graph: Graph, time_point_current: TimePoint, time_point_next: TimePoint, tolerance: float):
    """Adds edges to the next time point, which is useful if _add_edges missed some possible links."""
    for particle in time_point_current.particles():
        nearby_list = find_nearest_particles(time_point_next, particle, tolerance)
        for nearby_particle in nearby_list:
            if not graph.has_edge(particle, nearby_particle):
                graph.add_edge(particle, nearby_particle, pref=False)


def _all_links_downgraded(original_graph: Graph):
    """Returns a copy of the graph with all links with pref=True changed to pref=False"""
    graph = original_graph.copy()
    for particle1, particle2, data in graph.edges(data = True):
        if data["pref"]:
            data["pref"] = False  # This also modifies the graph
    return graph


def _find_nearest_edges_using_flow(graph: Graph, initial_links: Graph, time_point: TimePoint,
                                   flow_detection_radius: int):
    for particle in time_point.particles():
        possible_connections = _find_past_particles(graph, particle)
        if len(possible_connections) == 0:
            continue
        flow_x, flow_y, flow_z = particle_flow.get_flow_to_previous(initial_links, time_point, particle,
                                                                    max_dx_and_dy=flow_detection_radius)
        nearest = core.get_closest_particle(possible_connections,
                                            Particle(particle.x + flow_x, particle.y + flow_y, particle.z + flow_z))
        graph[particle][nearest]["pref"] = True


def _find_past_particles(graph: Graph, particle: Particle):
    try:
        all_connections = graph[particle]
    except KeyError:
        all_connections = []
    return {connection for connection in all_connections if connection.time_point_number() < particle.time_point_number()}