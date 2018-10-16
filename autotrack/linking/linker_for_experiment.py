from networkx import Graph

from autotrack.core.particles import Particle, get_closest_particle, ParticleCollection
from autotrack.core import TimePoint
from autotrack.core.experiment import Experiment
from autotrack.linking import particle_flow
from autotrack.linking.find_nearest_neighbors import find_nearest_particles


def nearest_neighbor(experiment: Experiment, tolerance: float = 1.0, over_previous: bool = False) -> Graph:
    """Simple nearest neighbour linking, keeping a list of potential candidates based on a given tolerance.

    A tolerance of 1.05 also links particles 5% from the closest particle. Note that if a tolerance higher than 1 is
    given, some pruning is needed on the final result.

    max_time_point is the last time point that will still be included.
    """
    graph = Graph()

    time_point_previous = None
    time_point_over_previous = None
    for time_point_current in experiment.time_points():
        _add_nodes(graph, experiment, time_point_current)

        if time_point_previous is not None:
            _add_nearest_edges(graph, experiment.particles, time_point_previous, time_point_current, tolerance)
            _add_nearest_edges_extra(graph, experiment.particles, time_point_previous, time_point_current, tolerance)
        if over_previous and time_point_over_previous is not None:
            _add_nearest_edges(graph, experiment.particles, time_point_over_previous, time_point_current, tolerance)
            _add_nearest_edges_extra(graph, experiment.particles, time_point_over_previous, time_point_current, tolerance)

        time_point_over_previous = time_point_previous
        time_point_previous = time_point_current

    print("Done creating nearest-neighbor links!")
    return graph


def nearest_neighbor_using_flow(experiment: Experiment, initial_links: Graph, flow_detection_radius: int) -> Graph:
    """A bit more advanced nearest-neighbor linking, that analysis the already-established links around a particle to
    decide in which direction a particle is going.

    max_time_point is the last time point that will still be included.
    """
    graph = _all_links_downgraded(initial_links)

    for time_point_current in experiment.time_points():
        _find_nearest_edges_using_flow(graph, initial_links, time_point_current, flow_detection_radius)

    print("Done creating nearest-neighbor links using flow!")
    return graph


def with_only_the_preferred_edges(old_graph: Graph):
    """Returns a new graph that only contains the links with a "pref" attribute."""
    graph = Graph()
    for node, data in old_graph.nodes(data=True):
        if not isinstance(node, Particle):
            raise ValueError("Found a node that was not a particle: " + str(node))
        graph.add_node(node, **data)

    for particle_1, particle_2, data in old_graph.edges(data=True):
        if data["pref"]:
            graph.add_edge(particle_1, particle_2)
    return graph


def _add_nodes(graph: Graph, experiment: Experiment, time_point: TimePoint) -> None:
    for particle in experiment.particles.of_time_point(time_point):
        graph.add_node(particle)


def _add_nearest_edges(graph: Graph, particles: ParticleCollection, time_point_previous: TimePoint, time_point_current: TimePoint, tolerance: float):
    """Adds edges pointing towards previous time point, making the shortest one the preferred."""
    for particle in particles.of_time_point(time_point_current):
        nearby_list = find_nearest_particles(particles, time_point_previous, particle, tolerance, max_amount=5)
        preferred = True
        for nearby_particle in nearby_list:
            graph.add_edge(particle, nearby_particle, pref=preferred)
            preferred = False  # All remaining links are not preferred


def _add_nearest_edges_extra(graph: Graph, particles: ParticleCollection, time_point_current: TimePoint, time_point_next: TimePoint, tolerance: float):
    """Adds edges to the next time point, which is useful if _add_edges missed some possible links."""
    for particle in particles.of_time_point(time_point_current):
        nearby_list = find_nearest_particles(particles, time_point_next, particle, tolerance, max_amount=5)
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


def _find_nearest_edges_using_flow(graph: Graph, particles: ParticleCollection, initial_links: Graph,
                                   time_point: TimePoint, flow_detection_radius: int):
    particles_of_time_point = particles.of_time_point(time_point)
    for particle in particles_of_time_point:
        possible_connections = _find_past_particles(graph, particle)
        if len(possible_connections) == 0:
            continue
        flow_x, flow_y, flow_z = particle_flow.get_flow_to_previous(initial_links, particles_of_time_point, particle,
                                                                    max_dx_and_dy=flow_detection_radius)
        nearest = get_closest_particle(possible_connections,
                                            Particle(particle.x + flow_x, particle.y + flow_y, particle.z + flow_z))
        graph[particle][nearest]["pref"] = True


def _find_past_particles(graph: Graph, particle: Particle):
    try:
        all_connections = graph[particle]
    except KeyError:
        all_connections = []
    return {connection for connection in all_connections if connection.time_point_number() < particle.time_point_number()}
