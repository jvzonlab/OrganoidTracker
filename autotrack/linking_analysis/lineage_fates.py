from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker


class LineageFate:
    """Calculates the number of occurences of certain events in the lineage."""

    divisions: int = 0  # How many divisions are there in the lineage?
    deaths: int = 0  # How many cell deaths are there in the lineage?
    errors: int = 0  # How many warnings are still in the lineage?
    ends: int = 0  # How many lineage ends (including cell deaths) are still in the lineage?


def get_lineage_fate(particle: Particle, links: ParticleLinks, last_time_point_number: int) -> LineageFate:
    """Calculates the fate of the lineage. The last time point number is used to ignore lineage ends that occur in that
    time point."""
    lineage_fate = LineageFate()
    _get_sub_cell_fate(particle, links, lineage_fate, last_time_point_number)
    return lineage_fate


def _get_sub_cell_fate(particle: Particle, links: ParticleLinks, lineage_fate: LineageFate, last_time_point_number: int):
    while True:
        error = linking_markers.get_error_marker(links, particle)
        if error is not None:
            lineage_fate.errors += 1

        next_particles = links.find_futures(particle)
        if len(next_particles) > 1:
            lineage_fate.divisions += 1
            for next_particle in next_particles:
                _get_sub_cell_fate(next_particle, links, lineage_fate, last_time_point_number)
            return
        elif len(next_particles) == 0:
            if particle.time_point_number() < last_time_point_number:
                lineage_fate.ends += 1  # Ignore lineage ends in the last time point
            if linking_markers.get_track_end_marker(links, particle) == EndMarker.DEAD:
                lineage_fate.deaths += 1
            return
        else:
            particle = next_particles.pop()
