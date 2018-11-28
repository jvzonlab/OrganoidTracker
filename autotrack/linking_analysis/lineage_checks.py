"""Used to find lineage trees with (potential) errors."""

from typing import List, Optional, Set, AbstractSet

from networkx import Graph

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.linking import existing_connections
from autotrack.linking_analysis import cell_appearance_finder, linking_markers


class LineageWithErrors:
    """Represents all (potential) errors in a  complete lineage tree."""

    start: Particle  # Start of the lineage tree
    errored_particles: List[Particle]  # All particles with (potential) errors
    crumbs: Set[Particle]  # Some of the particles in the lineage tree.

    def __init__(self, start: Particle):
        self.start = start
        self.errored_particles = []
        self.crumbs = set()


def get_problematic_lineages(links: ParticleLinks, crumbs: AbstractSet[Particle]) -> List[LineageWithErrors]:
    """Gets a list of all lineages with warnings in the experiment. The provided "crumbs" are placed in the right
    lineages, so that you can see to what lineages those cells belong."""
    lineages_with_errors = []
    for starting_cell in links.find_appeared_cells():
        lineage = LineageWithErrors(starting_cell)
        _find_errors_in_lineage(links, lineage, starting_cell, crumbs)
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
