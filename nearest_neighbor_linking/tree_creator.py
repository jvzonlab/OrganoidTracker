from networkx import Graph
from imaging import Experiment, Frame
from nearest_neighbor_linking.find_nearest_few import find_nearest_particles


def link_particles(experiment: Experiment, tolerance: float = 1.0, max_frame: int = 5000) -> Graph:
    """Simple nearest neighbour linking, keeping a list of potential candidates based on a given tolerance.

    A tolerance of 1.05 also links particles 5% from the closest particle. Note that if a tolerance higher than 1 is
    given, some pruning is needed on the final result.

    max_frame is the last frame that will still be included.
    """
    graph = Graph()

    frame_current = experiment.get_frame(1)
    _add_nodes(graph, frame_current)

    try:
        for i in range(0, max_frame - 1):
            frame_previous = frame_current

            frame_current = experiment.get_next_frame(frame_previous)
            _add_nodes(graph, frame_current)
            _add_edges(graph, frame_previous, frame_current, tolerance)
            print("Added links for frame " + str(frame_current.frame_number()))
    except KeyError:
        # Done! No more frames remain
        pass

    print("Done creating the tree graph!")
    return graph


def _add_nodes(graph: Graph, frame: Frame) -> None:
    for particle in frame.particles():
        graph.add_node(particle)


def _add_edges(graph: Graph, frame_previous: Frame, frame_current: Frame, tolerance: float):
    for particle in frame_current.particles():
        nearby_list = find_nearest_particles(frame_previous, particle, tolerance)
        preferred = True
        for nearby_particle in nearby_list:
            graph.add_edge(particle, nearby_particle, pref=preferred)
            preferred = False # All remaining links are not preferred
