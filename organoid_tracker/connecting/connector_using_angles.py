"""Uses a fancy algorithm to check whether a cell is in between two cells. If not, the cells are considered neighbors.
Original implementation by Max Betjes.

The idea is, that for every pair of cells that could be neighbors, we find the cell that is closes to the two. Then, we
check whether that cell is in between the two given cells. If no, then the two given cells are considered neighbors.
"""
import math
from typing import Optional

import networkx
from networkx import Graph

from organoid_tracker.core import TimePoint
from organoid_tracker.core.connections import Connections
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.linking import nearby_position_finder

_NEIGHBOR_SCORE_THRESHOLD = math.sqrt(2)


def _find_closest_common_neighbor(connections_graph: Graph, position_a: Position, position_b: Position) -> Optional[Position]:
    """Finds the nearest common neighbor C of position_a and position_b. Nearest is defined as having the smallest
    distance AC + BC."""
    smallest_distance_sum = float("inf")
    closest_common_neighbor = None

    for common_neighbor in networkx.common_neighbors(connections_graph, position_a, position_b):
        distance_sum = connections_graph[position_a][common_neighbor]["distance_um"]\
                       + connections_graph[position_b][common_neighbor]["distance_um"]

        if closest_common_neighbor is None or distance_sum < smallest_distance_sum:
            closest_common_neighbor = common_neighbor
            smallest_distance_sum = distance_sum

    return closest_common_neighbor


def _find_neighbors(experiment: Experiment, time_point: TimePoint, *, into: Connections):
    positions = list(experiment.positions.of_time_point(time_point))
    resolution = experiment.images.resolution()

    # Graph for which to calculate neighbor_scores
    small_connections_graph = nearby_position_finder.make_nearby_positions_graph(resolution, positions, neighbors=10)

    # All neighbors usable as cells in between pairs
    big_connections_graph = nearby_position_finder.make_nearby_positions_graph(resolution, positions, neighbors=20)

    # Find cells in between
    for position_a, position_b, distance_um in small_connections_graph.edges.data("distance_um"):
        between_position = _find_closest_common_neighbor(big_connections_graph, position_a, position_b)
        if between_position is None:
            continue  # No common neighbor
        neighbor_score = (big_connections_graph[position_a][between_position]["distance_um"]
                          + big_connections_graph[position_b][between_position]["distance_um"]) / distance_um
        if neighbor_score > _NEIGHBOR_SCORE_THRESHOLD:
            # Use this link!
            into.add_connection(position_a, position_b)


def create_connections(experiment: Experiment, *, print_progress: bool = False) -> Connections:
    """Creates the connections for the given experiment, using the relative distances."""
    connections = Connections()
    if print_progress:
        print(f"Working on {experiment.name}...")
    for time_point in experiment.positions.time_points():
        print(f"  Working on time point {time_point.time_point_number()}...")
        _find_neighbors(experiment, time_point, into=connections)
    return connections






