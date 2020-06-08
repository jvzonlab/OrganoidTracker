"""Assigns an id to an lineage tree. The ids are scrambled, so that cells next to each other don't have a similar id.
This makes it easier to color the cells using a color map.

Note: the color depends on the id. The id depends on the sort order of the lineages. Call the sorting method on the
links object beforehand to make the id better-defined."""
import random
from typing import Tuple

import matplotlib.cm, matplotlib.colors

from organoid_tracker.core import Color
from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position import Position

# Used for pseudo-randomization of the colors
_RANDOM = random.Random()


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

    return track_id


def generate_color_for_lineage_id(track_id: int) -> Color:
    """Gets the RGB color (three numbers from 0 to 1) that the given lineage tree should be drawn in. The id must be a track id or a
    lineage id."""
    if track_id == -1:
        return Color.black()  # Returns black for tracks without a lineage

    _RANDOM.seed(track_id ** 3)

    hue = _RANDOM.random()
    saturation = 0.3 + _RANDOM.random() * 0.7
    value = 0.3 + _RANDOM.random() * 0.7
    return Color.from_rgb_floats(*matplotlib.colors.hsv_to_rgb((hue, saturation, value)))


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
    # track is now the root of the lineage id

    if not in_a_tree:
        return -1

    track_id = links.get_track_id(track)

    # Randomize last three digits
    return track_id
