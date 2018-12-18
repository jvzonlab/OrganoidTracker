from autotrack.core.experiment import Experiment

from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker, StartMarker


def postprocess(experiment: Experiment, margin_xy: int):
    _remove_particles_close_to_edge(experiment, margin_xy)
    _remove_spurs(experiment)


def _remove_particles_close_to_edge(experiment: Experiment, margin_xy: int):
    image_loader = experiment.image_loader()
    links = experiment.links
    example_image = image_loader.get_image_stack(experiment.get_time_point(image_loader.first_time_point_number()))
    for time_point in experiment.time_points():
        for particle in list(experiment.particles.of_time_point(time_point)):
            if particle.x < margin_xy or particle.y < margin_xy or particle.x > example_image.shape[2] - margin_xy\
                    or particle.y > example_image.shape[1] - margin_xy:
                _add_out_of_view_markers(links, particle)
                experiment.remove_particle(particle)


def _add_out_of_view_markers(links: ParticleLinks, particle: Particle):
    """Adds markers to the remaining links so that it is clear why they appeared/disappeared."""
    linked_particles = links.find_links_of(particle)
    for linked_particle in linked_particles:
        if linked_particle.time_point_number() < particle.time_point_number():
            linking_markers.set_track_end_marker(links, linked_particle, EndMarker.OUT_OF_VIEW)
        else:
            linking_markers.set_track_start_marker(links, linked_particle, StartMarker.GOES_INTO_VIEW)


def _remove_spurs(experiment: Experiment):
    """Removes all very short tracks that end in a cell death."""
    links = experiment.links
    for particle in list(links.find_appeared_cells()):
        _check_for_and_remove_spur(experiment, links, particle)


def _check_for_and_remove_spur(experiment: Experiment, links: ParticleLinks, particle: Particle):
    track_length = 0
    particles_in_track = [particle]

    while True:
        next_particles = links.find_futures(particle)
        if len(next_particles) == 0:
            # End of track
            if track_length < 3:
                # Remove this track, it is too short
                for particle_in_track in particles_in_track:
                    experiment.remove_particle(particle_in_track)
            return
        if len(next_particles) > 1:
            # Cell division
            for next_particle in next_particles:
                _check_for_and_remove_spur(experiment, links, next_particle)
            return

        particle = next_particles.pop()
        particles_in_track.append(particle)
        track_length += 1
