from ai_track.core.experiment import Experiment

from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.linking_analysis import linking_markers
from ai_track.linking_analysis.linking_markers import EndMarker, StartMarker


def postprocess(experiment: Experiment, margin_xy: int):
    _remove_positions_close_to_edge(experiment, margin_xy)
    _remove_spurs(experiment)
    _mark_positions_going_out_of_image(experiment)


def _remove_positions_close_to_edge(experiment: Experiment, margin_xy: int):
    image_loader = experiment.images
    links = experiment.links
    for time_point in experiment.time_points():
        for position in list(experiment.positions.of_time_point(time_point)):
            if not image_loader.is_inside_image(position, margin_xy=margin_xy):
                # Remove cell, but inform neighbors first
                _add_out_of_view_markers(links, position)
                experiment.remove_position(position)


def _mark_positions_going_out_of_image(experiment: Experiment):
    """Adds "going into view" and "going out of view" markers to all positions that fall outside the next or previous
    image, in case the camera was moved."""
    for time_point in experiment.time_points():
        try:
            time_point_previous = experiment.get_previous_time_point(time_point)
        except ValueError:
            continue  # This is the first time point

        offset = experiment.images.offsets.of_time_point(time_point)
        offset_previous = experiment.images.offsets.of_time_point(time_point_previous)
        if offset == offset_previous:
            continue  # Image didn't move, so no positions can go out of the view

        for position in experiment.positions.of_time_point(time_point_previous):
            # Check for positions in the previous image that fall outside the current image
            if not experiment.images.is_inside_image(position.with_time_point(time_point)):
                linking_markers.set_track_end_marker(experiment.links, position, EndMarker.OUT_OF_VIEW)

        for position in experiment.positions.of_time_point(time_point):
            # Check for positions in the current image that fall outside the previous image
            if not experiment.images.is_inside_image(position.with_time_point(time_point_previous)):
                linking_markers.set_track_start_marker(experiment.links, position, StartMarker.GOES_INTO_VIEW)


def _add_out_of_view_markers(links: Links, position: Position):
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
    for position in list(links.find_appeared_positions()):
        _check_for_and_remove_spur(experiment, links, position)


def _check_for_and_remove_spur(experiment: Experiment, links: Links, position: Position):
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
