from typing import List, Optional

from autotrack.core.links import Links
from autotrack.core.positions import Position


def find_previous_positions(position: Position, links: Links, steps_back: int) -> Optional[List[Position]]:
    """Gets a list consisting of the given position and steps_back positions in previous time points. Returns None if
    we can't get back that many time points. Index 0 will be the given position, index 1 the position one time step back,
    etc."""
    position_list = [position]
    while position_list[0].time_point_number() - position.time_point_number() < steps_back:
        previous_set = links.find_pasts(position)
        if len(previous_set) == 0:
            return None
        position = previous_set.pop()
        position_list.append(position)
    return position_list
