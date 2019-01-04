from typing import Iterable, Dict

from numpy import ndarray

from autotrack.core.position import Position


def find_undetected_positions(labeled_image: ndarray, positions: Iterable[Position], image_offset: Position
                              ) -> Dict[Position, str]:
    """Returns a dict of position->error code for all positions that were undetected."""
    used_ids = dict()
    found_errors = dict()
    for position in positions:
        image_position = position.subtract_pos(image_offset)
        try:
            id = labeled_image[int(image_position.z), int(image_position.y), int(image_position.x)]
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
