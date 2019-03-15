from autotrack.core.links import LinkingTrack
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker
from autotrack.core.experiment import Experiment

def get_symmetry(links, track_1: LinkingTrack, track_2: LinkingTrack):
    cell_dead_1 = len(track_1.get_next_tracks())
    cell_dead_2 = len(track_2.get_next_tracks())
    cell_divide = len(track_1.get_next_tracks()) == 2 or len(track_1.get_next_tracks()) == 2
    """Returns True if symmetric (both divide or both don't divide), False otherwise."""
    if cell_dead_1 == cell_dead_2:
        # cells are dead, so it's symmetric
        return True
    else:
        #  One cell divides, other does not
        end_marker1 = linking_markers.get_track_end_marker(links, track_1.find_last_position())
        end_marker2 = linking_markers.get_track_end_marker(links, track_2.find_last_position())
        # max_time_point_number = track_1.time_point_number() + experiment.division_lookahead_time_points
        if end_marker1 == EndMarker.DEAD or end_marker2 == EndMarker.DEAD:
            # Cell died, other divided, so asymmetric
            return False
        # One of the cells went out of the view, so we can no longer track it
        # No idea if it's symmetric or not, but let's assume so
        return True