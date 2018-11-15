from collections import Iterable

from networkx import Graph

from autotrack.core.experiment import Experiment
from numpy import ndarray

from autotrack.core.particles import Particle
from autotrack.linking import existing_connections
from autotrack.linking_analysis import cell_appearance_finder, linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker, StartMarker


def postprocess(experiment: Experiment, margin_xy: int):
    _remove_particles_close_to_edge(experiment, margin_xy)
    _remove_spurs(experiment)


def _remove_particles_close_to_edge(experiment: Experiment, margin_xy: int):
    image_loader = experiment.image_loader()
    graph = experiment.links.graph
    example_image = image_loader.get_image_stack(experiment.get_time_point(image_loader.get_first_time_point()))
    for time_point in experiment.time_points():
        for particle in list(experiment.particles.of_time_point(time_point)):
            if particle.x < margin_xy or particle.y < margin_xy or particle.x > example_image.shape[2] - margin_xy\
                    or particle.y > example_image.shape[1] - margin_xy:
                _add_out_of_view_markers(graph, particle)
                experiment.remove_particle(particle)


def _add_out_of_view_markers(graph: Graph, particle: Particle):
    """Adds markers to the remaining links so that it is clear why they appeared/disappeared."""
    try:
        linked_particles: Iterable[Particle] = graph[particle]
        for linked_particle in linked_particles:
            if linked_particle.time_point_number() < particle.time_point_number():
                linking_markers.set_track_end_marker(graph, linked_particle, EndMarker.OUT_OF_VIEW)
            else:
                linking_markers.set_track_start_marker(graph, linked_particle, StartMarker.GOES_INTO_VIEW)
    except KeyError:
        pass  # Particle is not in linking network


def _remove_spurs(experiment: Experiment):
    """Removes all very short tracks that end in a cell death."""
    graph = experiment.links.graph
    for particle in list(cell_appearance_finder.find_appeared_cells(graph)):
        _check_for_and_remove_spur(experiment, graph, particle)


def _check_for_and_remove_spur(experiment: Experiment, graph: Graph, particle: Particle):
    track_length = 0
    particles_in_track = [particle]

    while True:
        next_particles = existing_connections.find_future_particles(graph, particle)
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
                _check_for_and_remove_spur(experiment, graph, next_particle)
            return

        particle = next_particles.pop()
        particles_in_track.append(particle)
        track_length += 1
