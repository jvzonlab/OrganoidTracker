from typing import Tuple, Iterable

import numpy

from autotrack.core.links import Links
from autotrack.core.positions import Position


def get_flow_to_previous(links: Links, positions: Iterable[Position], center: Position,
                         max_dx_and_dy: int = 50, max_dz = 2) -> Tuple[float, float, float]:
    """Gets the average flow of the positions within the specified radius towards the previous time point. Returns
    (0,0,0) if there are no positions. The given center position must be in the givne time point.
    """

    count = 0
    total_movement = numpy.zeros(3)
    for position in positions:
        if _is_far_way_or_same(center, position, max_dx_and_dy, max_dz):
            continue

        past_positions = links.find_pasts(position)
        if len(past_positions) != 1:
            continue

        past_position = past_positions.pop()
        total_movement += (past_position.x - position.x, past_position.y - position.y, past_position.z - position.z)
        count += 1

    if count == 0:
        return 0, 0, 0
    return total_movement[0] / count, total_movement[1] / count, total_movement[2] / count


def get_flow_to_next(links: Links, positions: Iterable[Position], center: Position,
                     max_dx_and_dy: int = 50, max_dz = 2) -> Tuple[float, float, float]:
    """Gets the average flow of the positions within the specified radius towards the next time point. Returns
    (0,0,0) if there are no positions. The given center position must be in the given time point. Ignores cell
    divisions and dead cells.
    """

    count = 0
    total_movement = numpy.zeros(3)
    for position in positions:
        if _is_far_way_or_same(center, position, max_dx_and_dy, max_dz):
            continue

        next_positions = links.find_futures(position)
        if len(next_positions) != 1:
            continue  # Cell division or dead cell; ignore
        next_position = next_positions.pop()

        total_movement += (next_position.x - position.x, next_position.y - position.y, next_position.z - position.z)
        count += 1

    if count == 0:
        return (0, 0, 0)
    return total_movement[0] / count, total_movement[1] / count, total_movement[2] / count


def _is_far_way_or_same(center: Position, position: Position, max_dx_and_dy: int, max_dz: int) -> bool:
    if position == center:
        return True  # The center is ignored
    if abs(position.x - center.x) > max_dx_and_dy:
        return True
    if abs(position.y - center.y) > max_dx_and_dy:
        return True
    if abs(position.z - center.z) > max_dz:
        return True
    return False
