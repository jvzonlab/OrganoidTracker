"""A quick and dirty way to measure curvature of cells. Identify the nearest N positions around a position P. For every
nearby position (called NP), look at exactly the opposite side of P for another position (called AP). The curvature is
then the average angle NP-P-AP. In a perfectly flat hexagonal lattice, this value will be 180 degrees."""
from typing import List, Tuple

from autotrack.core.experiment import Experiment
from autotrack.core.position import Position
from autotrack.imaging import angles
from autotrack.linking import nearby_position_finder


_NEARBY_CELLS_AMOUNT = 6


def get_curvature_pairs(experiment: Experiment, position: Position) -> List[Tuple[Position, Position]]:
    """Gets the "curvateure pairs" NP-AP (see comments on this module). Returned list is in the format
    [(NP, AP), (NP, AP), ...]."""
    resolution = experiment.images.resolution()
    other_positions = experiment.positions.of_time_point(position.time_point())
    positions_around = nearby_position_finder.find_closest_n_positions(other_positions, around=position,
                                                                       max_amount=_NEARBY_CELLS_AMOUNT,
                                                                       resolution=resolution)

    pairs_around = list()
    for position_around in positions_around:
        delta = position - position_around
        opposite_coord = position + delta
        positions_around_without_current = positions_around.difference({position_around})
        opposite_position = nearby_position_finder.find_closest_position(positions_around_without_current,
                                                                         around=opposite_coord, resolution=resolution)
        pairs_around.append((position_around, opposite_position))

    return pairs_around


def get_curvature_angle(experiment: Experiment, position: Position) -> float:
    """Gets the average angle between the "curvature pairs" NP-AP (see comments on this module)."""
    resolution = experiment.images.resolution()
    found_angles = list()
    for position1, position2 in get_curvature_pairs(experiment, position):
        found_angle = angles.right_hand_rule(position1.to_vector_um(resolution), position.to_vector_um(resolution),
                                             position2.to_vector_um(resolution))
        found_angles.append(found_angle)
    return sum(found_angles) / len(found_angles)
