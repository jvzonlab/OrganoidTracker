"""Used to find divisions in linking data."""

import itertools
from typing import Set, List, Optional

from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.score import Family


def get_next_division(links: Links, position: Position) -> Optional[Family]:
    """Gets the next division for the given position. Returns None if there is no such division. Raises ValueError if a
    cell with more than two daughters is found in this lineage."""
    track = links.get_track(position)
    if track is None:
        return None

    next_tracks = track.get_next_tracks()
    if len(next_tracks) < 2:
        return None

    next_daughters = [next_track.find_first_position() for next_track in next_tracks]
    if len(next_daughters) != 2:
        raise ValueError("Cell " + str(track.find_last_position()) + " has multiple daughters: " + str(next_daughters))

    return Family(track.find_last_position(), next_daughters[0], next_daughters[1])


def get_previous_division(links: Links, position: Position) -> Optional[Family]:
    """Finds the previous division of a daughter cell. Returns None if the cell is not a daughter cell."""
    track = links.get_track(position)
    if track is None:
        # Position is not part of a track, so it has no links, so there is no previous division
        return None

    previous_tracks = track.get_previous_tracks()
    if len(previous_tracks) == 0:
        return None  # No previous track, cell appeared out of nothing
    if len(previous_tracks) > 1:
        print(f"Cell {track.find_first_position()} has multiple links to the past")
        return None

    previous_track = previous_tracks.pop()
    sibling_tracks = previous_track.get_next_tracks()
    if len(sibling_tracks) < 2:
        return None  # No division here

    siblings = [sibling_track.find_first_position() for sibling_track in sibling_tracks]
    if len(siblings) != 2:
        raise ValueError("Cell " + str(previous_track.find_last_position()) + " has multiple daughters: " + str(siblings))

    return Family(previous_track.find_last_position(), siblings[0], siblings[1])


def find_mothers(links: Links, exclude_multipolar = True) -> Set[Position]:
    """Finds all mother cells in a graph. Mother cells are cells with at least two daughter cells."""
    mothers = set()

    for track in links.find_all_tracks():
        future_tracks = track.get_next_tracks()
        if len(future_tracks) >= 2:
            mothers.add(track.find_last_position())
        if len(future_tracks) > 2:
            if exclude_multipolar:
                print("Illegal mother: " + str(len(future_tracks)) + " daughters found")
            else:
                mothers.add(track.find_last_position())

    return mothers

def find_families(links: Links, warn_on_many_daughters = True) -> List[Family]:
    """Finds all mother and daughter cells in a graph. Mother cells are cells with at least two daughter cells.
    Returns a set of Family instances.
    """
    families = list()

    for track in links.find_all_tracks():
        next_tracks = track.get_next_tracks()
        if len(next_tracks) < 2:
            continue
        next_daughters = [next_track.find_first_position() for next_track in next_tracks]
        if warn_on_many_daughters:
            # Only two daughters are allowed
            families.append(Family(track.find_last_position(), next_daughters[0], next_daughters[1]))
            if len(next_tracks) > 2:
                print("Illegal mother: " + str(len(next_tracks)) + " daughters found")
        else:
            # Many daughters are allowed
            for daughter1, daughter2 in itertools.combinations(next_daughters, 2):
                families.append(Family(track.find_last_position(), daughter1, daughter2))

    return families
