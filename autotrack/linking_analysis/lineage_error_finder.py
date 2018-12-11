"""Used to find lineage trees with (potential) errors."""

from typing import List, Optional, Set, AbstractSet, Dict, Iterable

from autotrack.core.links import ParticleLinks, LinkingTrack
from autotrack.core.particles import Particle
from autotrack.linking_analysis import linking_markers


class LineageWithErrors:
    """Represents all (potential) errors in a  complete lineage tree."""

    start: LinkingTrack  # Start of the lineage tree
    errored_particles: List[Particle]  # All particles with (potential) errors
    crumbs: Set[Particle]  # Some of the particles in the lineage tree.

    def __init__(self, start: LinkingTrack):
        self.start = start
        self.errored_particles = []
        self.crumbs = set()

    def _add_errors(self, errors: Optional[Iterable[Particle]]):
        if errors is not None:
            self.errored_particles += errors

    def _add_crumbs(self, crumbs: Optional[Iterable[Particle]]):
        if crumbs is not None:
            self.crumbs |= set(crumbs)


def _group_by_track(links: ParticleLinks, particles: Iterable[Particle]) -> Dict[LinkingTrack, List[Particle]]:
    track_to_particles = dict()
    for particle in particles:
        track = links.get_track(particle)
        if track in track_to_particles:
            track_to_particles[track].append(particle)
        else:
            track_to_particles[track] = [particle]
    return track_to_particles


def get_problematic_lineages(links: ParticleLinks, crumbs: AbstractSet[Particle]) -> List[LineageWithErrors]:
    """Gets a list of all lineages with warnings in the experiment. The provided "crumbs" are placed in the right
    lineages, so that you can see to what lineages those cells belong."""
    particles_with_errors = linking_markers.find_errored_particles(links)
    track_to_errors = _group_by_track(links, particles_with_errors)
    track_to_crumbs = _group_by_track(links, crumbs)

    lineages_with_errors = list()
    for track in links.find_starting_tracks():
        lineage = LineageWithErrors(track)
        lineage._add_errors(track_to_errors.get(track))
        lineage._add_crumbs(track_to_crumbs.get(track))

        for next_track in track.find_all_descending_tracks():
            lineage._add_errors(track_to_errors.get(next_track))
            lineage._add_crumbs(track_to_crumbs.get(next_track))

        if len(lineage.errored_particles) > 0:
            lineages_with_errors.append(lineage)

    return lineages_with_errors


def _find_errors_in_lineage(links: ParticleLinks, lineage: LineageWithErrors, particle: Particle, crumbs: AbstractSet[Particle]):
    while True:
        if particle in crumbs:
            lineage.crumbs.add(particle)

        error = linking_markers.get_error_marker(links, particle)
        if error is not None:
            lineage.errored_particles.append(particle)
        future_particles = links.find_futures(particle)

        if len(future_particles) > 1:
            # Branch out
            for future_particle in future_particles:
                _find_errors_in_lineage(links, lineage, future_particle, crumbs)
            return
        if len(future_particles) < 1:
            # Stop
            return
        # Continue
        particle = future_particles.pop()


def find_lineage_index_with_crumb(lineages: List[LineageWithErrors], crumb: Optional[Particle]) -> Optional[int]:
    """Attempts to find the given particle in the lineages. Returns None if the particle is None or if the particle is
    in none of the lineages."""
    if crumb is None:
        return None
    for index, lineage in enumerate(lineages):
        if crumb in lineage.crumbs:
            return index
    return None
