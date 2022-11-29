"""Methods for working with projecting splines on the sphere."""
from typing import Iterable

import matplotlib.cm
from matplotlib.colors import Colormap

from organoid_tracker.coordinate_system.sphere_representer import SphereRepresentation
from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack, Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.spline import SplineCollection


def add_all_splines(sphere_representation: SphereRepresentation, splines: SplineCollection):
    """Adds all splines in the given collection to the sphere as orientation markers."""
    time_points = list(splines.time_points())
    time_points.reverse()
    for spline_id, _ in splines.of_time_point(TimePoint(splines.last_time_point_number())):
        past_crypt_positions = list()
        for time_point in time_points:
            spline = splines.get_spline(time_point, spline_id)
            if spline is None:
                continue
            crypt_bottom = spline.from_position_on_axis(0)
            if crypt_bottom is not None:
                crypt_bottom_position = Position(crypt_bottom[0], crypt_bottom[1], crypt_bottom[2],
                                                 time_point=time_point)
                past_crypt_positions.append(crypt_bottom_position)
        sphere_representation.add_orientation_track(past_crypt_positions)


class ColoredTrackAdder:
    """Adds linking tracks to the sphere. Tracks will be colored by their position along the closest spline."""

    _experiment: Experiment
    _largest_spline_length: float
    color_map: Colormap

    def __init__(self, experiment: Experiment):
        self._experiment = experiment
        self._largest_spline_length = self._calculate_longest_spline_length()
        # noinspection PyUnresolvedReferences
        self.color_map = matplotlib.cm.gist_stern

    def _calculate_longest_spline_length(self) -> float:
        last_time_point_number = self._experiment.splines.last_time_point_number()
        if last_time_point_number is None:
            raise UserError("No splines found", "No splines have been drawn. These are used to locate the crypt.")
        max_length = 1
        for spline_index, spline in self._experiment.splines.of_time_point(TimePoint(last_time_point_number)):
            max_length = max(max_length, spline.length())
        return max_length

    def add_track_colored_by_spline_position(self, sphere_representation: SphereRepresentation,
                                             linking_track: Iterable[Position], *, highlight_first: bool = False,
                                             highlight_last: bool = False):
        """Adds the given track to the sphere. The track will be colored by the position along the splines that were
        drawn."""
        position_list = list()
        color_list = list()
        for position in linking_track:
            position_list.append(position)
            pos_on_spline = self._experiment.splines.to_position_on_original_axis(self._experiment.links, position)
            if pos_on_spline is None:
                color_list.append("black")
            else:
                pos_on_spline_fraction = min(1, pos_on_spline.pos / self._largest_spline_length)
                color_list.append(self.color_map(1 - pos_on_spline_fraction))
        sphere_representation.add_track(position_list, colors=color_list, highlight_first=highlight_first,
                                        highlight_last=highlight_last)

