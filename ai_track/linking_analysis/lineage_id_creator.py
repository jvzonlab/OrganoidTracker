"""Assigns an id to an lineage tree."""
from ai_track.core.links import Links
from ai_track.core.position import Position


def get_lineage_id(links: Links, position: Position) -> int:
    """If the given cell is in a lineage tree (i.e. the cell has previously divided, or will divide during the
    experiment) this returns a unique id for that lineage tree. If not, then -1 is returned."""
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
    return links.get_track_id(track)
