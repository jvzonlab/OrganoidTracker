from networkx import Graph
from nearest_neighbor_linking.positions import Particle, Experiment, Frame
from nearest_neighbor_linking.find_nearest_few import find_nearest_particles


def link_particles(experiment: Experiment, tolerance: float) -> Graph:
    graph = Graph()

    frame_previous = experiment.get_frame(1)
    _add_nodes(graph, frame_previous)
    frame_current = experiment.get_next_frame(frame_previous)
    _add_nodes(graph, frame_current)
    _add_edges(graph, frame_previous, frame_current, tolerance)

    return graph


def _add_nodes(graph: Graph, frame: Frame) -> None:
    for particle in frame.particles():
        graph.add_node(particle)


def _add_edges(graph: Graph, frame_previous: Frame, frame_current: Frame, tolerance: float):
    for particle in frame_current.particles():
        nearby_list = find_nearest_particles(frame_previous, particle, tolerance)
        for nearby_particle in nearby_list:
            graph.add_edge(particle, nearby_particle)

