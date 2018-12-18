from typing import Iterable, Dict

from numpy import ndarray

from autotrack.core.positions import Position


def find_undetected_positions(labeled_image: ndarray, positions: Iterable[Position]) -> Dict[Position, str]:
    """Returns a dict of position->error code for all positions that were undetected."""
    used_ids = dict()
    found_errors = dict()
    for position in positions:
        try:
            id = labeled_image[int(position.z), int(position.y), int(position.x)]
            if id == 0:
                found_errors[position] = "Missed"
                continue
            if id in used_ids:
                found_errors[position] = "Merged with " + str(used_ids[id])
                continue
            used_ids[id] = position
        except IndexError:
            found_errors[position] = "Outside image"
    return found_errors
