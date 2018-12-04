"""Extra markers used to describe the linking data. For example, you can mark the end of a lineage as a cell death."""

from enum import Enum
from typing import Optional, Dict, Iterable

from networkx import Graph

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.linking_analysis.errors import Error


class EndMarker(Enum):
    DEAD = 1
    OUT_OF_VIEW = 2

    def get_display_name(self):
        """Gets a user-friendly display name."""
        return self.name.lower().replace("_", " ")


class StartMarker(Enum):
    GOES_INTO_VIEW = 1
    UNSURE = 2

    def get_display_name(self):
        """Gets a user-friendly display name."""
        return self.name.lower().replace("_", " ")


def get_track_end_marker(links: ParticleLinks, particle: Particle) -> Optional[EndMarker]:
    """Gets a death marker, which provides a reason why the cell lineage ended."""
    ending_str = links.get_particle_data(particle, "ending")
    if ending_str is None:
        return None

    return EndMarker[ending_str.upper()]


def set_track_end_marker(links: ParticleLinks, particle: Particle, end_marker: Optional[EndMarker]):
    """Sets a reason why the track ended at the given point."""
    if end_marker is None:
        links.set_particle_data(particle, "ending", None)
    else:
        links.set_particle_data(particle, "ending", end_marker.name.lower())


def find_dead_particles(links: ParticleLinks) -> Iterable[Particle]:
    """Gets all particles that were marked as dead."""
    death_marker = EndMarker.DEAD.name.lower()
    for particle, ending_marker in links.find_all_particles_with_data("ending"):
        if ending_marker == death_marker:
            yield particle


def get_track_start_marker(links: ParticleLinks, particle: Particle) -> Optional[StartMarker]:
    """Gets the appearance marker. This is used to explain why a cell appeared out of thin air."""
    starting_str = links.get_particle_data(particle, "starting")
    if starting_str is None:
        return None

    return StartMarker[starting_str.upper()]


def set_track_start_marker(links: ParticleLinks, particle: Particle, start_marker: Optional[StartMarker]):
    """Sets a reason why the track ended at the given point."""
    if start_marker is None:
        links.set_particle_data(particle, "starting", None)
    else:
        links.set_particle_data(particle, "starting", start_marker.name.lower())


def get_errored_particles(links: ParticleLinks) -> Iterable[Particle]:
    """Gets all particles that have a (non suppressed) error."""

    with_error_marker = links.find_all_particles_with_data("error")
    for particle, error_number in with_error_marker:
        if links.get_particle_data(particle, "suppressed_error") == error_number:
            continue # Error was suppressed

        yield particle


def get_error_marker(links: ParticleLinks, particle: Particle) -> Optional[Error]:
    """Gets the error marker for the given link, if any. Returns None if the error has been suppressed using
    suppress_error_marker."""
    error_number = links.get_particle_data(particle, "error")
    if error_number is None:
        return None

    if links.get_particle_data(particle, "suppressed_error") == error_number:
        return None  # Error was suppressed
    return Error(error_number)


def suppress_error_marker(links: ParticleLinks, particle: Particle, error: Error):
    """Suppresses an error. Even if set_error_marker is called afterwards, the error will not show up in
    get_error_marker."""
    links.set_particle_data(particle, "suppressed_error", error.value)


def is_error_suppressed(links: ParticleLinks, particle: Particle, error: Error) -> bool:
    """Returns True if the given error is suppressed. If another type of error is suppressed, this method returns
    False."""
    return links.get_particle_data(particle, "suppressed_error") == error.value


def set_error_marker(links: ParticleLinks, particle: Particle, error: Optional[Error]):
    """Sets an error marker for the given particle."""
    if error is None:
        links.set_particle_data(particle, "error", None)
    else:
        links.set_particle_data(particle, "error", error.value)
