"""Extra markers used to describe the linking data. For example, you can mark the end of a lineage as a cell death."""

from enum import Enum
from typing import Optional

from networkx import Graph

from autotrack.core.particles import Particle
from autotrack.linking_analysis.errors import Error


class EndMarker(Enum):
    DEAD = 1
    OUT_OF_VIEW = 2
    UNSURE = 3

    def get_display_name(self):
        """Gets a user-friendly display name."""
        return self.name.lower().replace("_", " ")


class StartMarker(Enum):
    GOES_INTO_VIEW = 1
    UNSURE = 2

    def get_display_name(self):
        """Gets a user-friendly display name."""
        return self.name.lower().replace("_", " ")


def get_track_end_marker(links: Graph, particle: Particle) -> Optional[EndMarker]:
    """Gets a death marker, which provides a reason why the cell lineage ended."""
    node_data = links.nodes.get(particle)
    if node_data is None:
        return None

    try:
        ending_str: str = node_data["ending"]
        return EndMarker[ending_str.upper()]
    except KeyError:
        return None


def set_track_end_marker(links: Graph, particle: Particle, end_marker: Optional[EndMarker]):
    """Sets a reason why the track ended at the given point."""
    if end_marker is None:
        try:
            del links.nodes[particle]["ending"]
        except KeyError:
            pass  # Ignore, nothing to delete
    else:
        links.nodes[particle]["ending"] = end_marker.name.lower()


def get_track_start_marker(links: Graph, particle: Particle) -> Optional[StartMarker]:
    """Gets the appearance marker. This is used to explain why a cell appeared out of thin air."""
    node_data = links.nodes.get(particle)
    if node_data is None:
        return None

    try:
        starting_str: str = node_data["starting"]
        return StartMarker[starting_str.upper()]
    except KeyError:
        return None


def set_track_start_marker(links: Graph, particle: Particle, start_marker: Optional[StartMarker]):
    """Sets a reason why the track ended at the given point."""
    if start_marker is None:
        try:
            del links.nodes[particle]["starting"]
        except KeyError:
            pass  # Ignore, nothing to delete
    else:
        links.nodes[particle]["starting"] = start_marker.name.lower()


def get_error_marker(links: Graph, particle: Particle) -> Optional[Error]:
    """Gets the error marker for the given link, if any."""
    node_data = links.nodes.get(particle)
    if node_data is None:
        return None

    if "error" not in node_data:
        return None

    error = node_data["error"]
    if error is None:
        return None  # In the past, we used to store None to delete errors

    return Error(error)


def set_error_marker(links: Graph, particle: Particle, error: Optional[Error]):
    """Sets an error marker for the given particle."""
    if error is None:
        if "error" in links.nodes[particle]:
            del links.nodes[particle]["error"]
    else:
        links.nodes[particle]["error"] = error.value
