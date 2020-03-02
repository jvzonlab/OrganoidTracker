from enum import Enum

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.linking import nearby_position_finder

_NEIGHBOR_COUNT = 6


class CellCompartment(Enum):
    DIVIDING = 0
    NON_DIVIDING = 1
    UNKNOWN = 2


def _will_divide(links: Links, position: Position) -> bool:
    """A quick check to see if a cell will certainly divide in the future. Returns False if no division was detected."""
    track = links.get_track(position)
    if track is None:
        # No links found for this position
        return False
    return len(track.get_next_tracks()) >= 2


def find_compartment(experiment: Experiment, position: Position) -> CellCompartment:
    """Finds the compartment of the cell. If a cell is non-dividing, but a cell close by is dividing, then the cell is
    considered to be in a dividing compartment."""
    if experiment.last_time_point_number() - position.time_point_number() < experiment.division_lookahead_time_points:
        return CellCompartment.UNKNOWN

    links = experiment.links
    if _will_divide(links, position):
        return CellCompartment.DIVIDING  # Cell will divide, so surely part of dividing compartment

    for nearby_position in nearby_position_finder.find_closest_n_positions(experiment.positions.of_time_point(
            position.time_point()), around=position, max_amount=_NEIGHBOR_COUNT,
            resolution=experiment.images.resolution()):
        if _will_divide(links, nearby_position):
            return CellCompartment.DIVIDING  # Neighbor cell will divide, so surely part of dividing compartment

    # No dividing cells nearby, assume non-dividing
    return CellCompartment.NON_DIVIDING

