from autotrack.core.experiment import Experiment

from autotrack.core.links import PositionLinks
from autotrack.core.positions import Position
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker, StartMarker


def postprocess(experiment: Experiment, margin_xy: int):
    _remove_positions_close_to_edge(experiment, margin_xy)
    _remove_spurs(experiment)


def _remove_positions_close_to_edge(experiment: Experiment, margin_xy: int):
    image_loader = experiment.image_loader()
    links = experiment.links
    example_image = image_loader.get_image_stack(experiment.get_time_point(image_loader.first_time_point_number()))
    for time_point in experiment.time_points():
        for position in list(experiment.positions.of_time_point(time_point)):
            if position.x < margin_xy or position.y < margin_xy or position.x > example_image.shape[2] - margin_xy\
                    or position.y > example_image.shape[1] - margin_xy:
                _add_out_of_view_markers(links, position)
                experiment.remove_position(position)


def _add_out_of_view_markers(links: PositionLinks, position: Position):
    """Adds markers to the remaining links so that it is clear why they appeared/disappeared."""
    linked_positions = links.find_links_of(position)
    for linked_position in linked_positions:
        if linked_position.time_point_number() < position.time_point_number():
            linking_markers.set_track_end_marker(links, linked_position, EndMarker.OUT_OF_VIEW)
        else:
            linking_markers.set_track_start_marker(links, linked_position, StartMarker.GOES_INTO_VIEW)


def _remove_spurs(experiment: Experiment):
    """Removes all very short tracks that end in a cell death."""
    links = experiment.links
    for position in list(links.find_appeared_cells()):
        _check_for_and_remove_spur(experiment, links, position)


def _check_for_and_remove_spur(experiment: Experiment, links: PositionLinks, position: Position):
    track_length = 0
    positions_in_track = [position]

    while True:
        next_positions = links.find_futures(position)
        if len(next_positions) == 0:
            # End of track
            if track_length < 3:
                # Remove this track, it is too short
                for position_in_track in positions_in_track:
                    experiment.remove_position(position_in_track)
            return
        if len(next_positions) > 1:
            # Cell division
            for next_position in next_positions:
                _check_for_and_remove_spur(experiment, links, next_position)
            return

        position = next_positions.pop()
        positions_in_track.append(position)
        track_length += 1
