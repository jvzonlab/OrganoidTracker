"""Assigns an id to an lineage tree. The ids are scrambled, so that cells next to each other don't have a similar id.
This makes it easier to color the cells using a color map."""
import random
from typing import Dict

import numpy

from ai_track.core.links import Links, LinkingTrack
from ai_track.core.position import Position

# Used for randomization of last three digits
_RANDOM_LIST = list(range(2000))
random.Random(12).shuffle(_RANDOM_LIST)


def get_original_track_id(links: Links, position: Position) -> int:
    """Gets a scrambled id for the original track the position was in. Unlike get_lineage_id, this function returns a
    value for cells that have never divided."""
    track = links.get_track(position)
    if track is None:
        return -1

    # Find original track
    previous_tracks = track.get_previous_tracks()
    while len(previous_tracks) == 1:
        track = previous_tracks.pop()
        previous_tracks = track.get_previous_tracks()

    track_id = links.get_track_id(track)

    # Randomize last three digits
    track_id_randomized = (track_id // len(_RANDOM_LIST)) * len(_RANDOM_LIST) \
                          + _RANDOM_LIST[track_id % len(_RANDOM_LIST)]

    return track_id_randomized


def get_lineage_id(links: Links, position: Position) -> int:
    """If the given cell is in a lineage tree (i.e. the cell has previously divided, or will divide during the
    experiment) this returns a scrambled id for that lineage tree. So for cells that do have tracks, but have never
    divided, this method returns -1."""
    track = links.get_track(position)
    if track is None:
        return -1

    in_a_tree = len(track.get_next_tracks()) > 1  # If there is a division after this, there is a lineage tree

    previous_tracks = track.get_previous_tracks()
    while len(previous_tracks) == 1:
        in_a_tree = True  # Found a previous track, so definitely a lineage tree
        track = previous_tracks.pop()
        previous_tracks = track.get_previous_tracks()

    if not in_a_tree:
        return -1

    track_id = links.get_track_id(track)

    # Randomize last three digits
    track_id_randomized = (track_id // len(_RANDOM_LIST)) * len(_RANDOM_LIST) \
                          + _RANDOM_LIST[track_id % len(_RANDOM_LIST)]

    return track_id_randomized
