from typing import Optional

from autotrack.core.experiment import Experiment
from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.resolution import ImageResolution
from autotrack.core.score import Score, ScoreCollection, Family
from autotrack.linking_analysis import linking_markers, particle_age_finder
from autotrack.linking_analysis.errors import Error
from autotrack.linking_analysis.linking_markers import EndMarker


def apply(experiment: Experiment):
    """Adds errors for all logical inconsistencies in the graph, like cells that spawn out of nowhere, cells that
    merge together and cells that have three or more daughters."""
    links = experiment.links
    scores = experiment.scores
    particles = experiment.particles
    resolution = experiment.image_resolution()
    for particle in links.find_all_particles():
        error = get_error(links, particle, scores, particles, resolution)
        linking_markers.set_error_marker(links, particle, error)


def get_error(links: ParticleLinks, particle: Particle, scores: ScoreCollection, particles: ParticleCollection,
              resolution: ImageResolution) -> Optional[Error]:
    future_particles = links.find_futures(particle)
    if len(future_particles) > 2:
        return Error.TOO_MANY_DAUGHTER_CELLS
    elif len(future_particles) == 0 \
            and particle.time_point_number() < particles.last_time_point_number() \
            and linking_markers.get_track_end_marker(links, particle) is None:
        return Error.NO_FUTURE_POSITION
    elif len(future_particles) == 2:
        if scores.has_scores():
            score = scores.of_family(Family(particle, *future_particles))
            if score is None or score.is_unlikely_mother():
                return Error.LOW_MOTHER_SCORE
        age = particle_age_finder.get_age(links, particle)
        if age is not None and age < 5:
            return Error.YOUNG_MOTHER

    past_particles = links.find_pasts(particle)
    if len(past_particles) == 0:
        if particle.time_point_number() > particles.first_time_point_number() \
                and linking_markers.get_track_start_marker(links, particle) is None:
            return Error.NO_PAST_POSITION
    elif len(past_particles) >= 2:
        return Error.CELL_MERGE
    else:  # len(past_particles) == 1
        # Check cell size
        past_particle = past_particles.pop()
        future_particles_of_past_particle = links.find_futures(past_particle)
        shape = particles.get_shape(particle)
        if not shape.is_unknown() and len(future_particles_of_past_particle) == 1:
            past_shape = particles.get_shape(past_particle)
            if not past_shape.is_unknown() and past_shape.volume() / (shape.volume() + 0.0001) > 3:
                return Error.SHRUNK_A_LOT

        # Check movement distance (fast movement is only allowed when a cell is launched into its death)
        if past_particle.distance_um(particle, resolution) > 10 and \
                linking_markers.get_track_end_marker(links, particle) != EndMarker.DEAD:
            return Error.MOVED_TOO_FAST
    return None


def _get_highest_mother_score(scores: ScoreCollection, particle: Particle) -> Optional[Score]:
    highest_score = None
    highest_score_num = -999
    for scored_family in scores.of_time_point(particle.time_point()):
        score = scored_family.score
        score_num = score.total()
        if score_num > highest_score_num:
            highest_score = score
            highest_score_num = score_num
    return highest_score


def apply_on(experiment: Experiment, *iterable: Particle):
    """Adds errors for all logical inconsistencies for particles in the collection, like cells that spawn out of
    nowhere, cells that merge together and cells that have three or more daughters."""
    links = experiment.links
    for particle in iterable:
        error = get_error(links, particle, experiment.scores, experiment.particles, experiment.image_resolution())
        linking_markers.set_error_marker(links, particle, error)
