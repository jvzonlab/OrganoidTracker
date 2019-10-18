"""High-level API for handling the lineage metadata."""
from typing import Optional

from ai_track.core import Color
from ai_track.core.links import LinkingTrack, Links


def set_color(links: Links, track: LinkingTrack, color: Color):
    """Sets the given lineage to the given color."""
    if color.is_black():
        links.set_lineage_data(track, "color", None)
    else:
        links.set_lineage_data(track, "color", color.to_rgb())


def get_color(links: Links, track: LinkingTrack) -> Color:
    """Gets the color of the given lineage."""
    color_number = links.get_lineage_data(track, "color")
    if color_number is None:
        return Color.black()
    return Color.from_rgb(color_number)
