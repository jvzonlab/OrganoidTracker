import math
from typing import NamedTuple, Optional

import numpy
from scipy.stats import linregress

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.position_analysis import position_markers


class IntensityOverTime(NamedTuple):
    """Represents an intensity over time."""
    mean: float
    mean_stderr: float
    slope: float
    slope_stderr: float


def get_intensity_derivative(experiment: Experiment, around_position: Position, time_window_h: float
                             ) -> Optional[IntensityOverTime]:
    """Gets the slope and mean intensity over time given time span. Returns None if not enough data is
    available, which is the case if the track is too short or if some intensities are missing."""
    resolution = experiment.images.resolution()
    time_window_tp = int(math.ceil(time_window_h / resolution.time_point_interval_h))
    tracking_start_time_point = int(around_position.time_point_number() - time_window_tp / 2)
    tracking_end_time_point = int(math.ceil(around_position.time_point_number() + time_window_tp / 2))
    if tracking_end_time_point - tracking_start_time_point < 5:
        raise ValueError("Time point window of " + str(time_window_h) + "h is too small")

    # Find the track
    track = experiment.links.get_track(around_position)
    if track is None:
        return None
    if track.min_time_point_number() > tracking_start_time_point\
            or track.max_time_point_number() < tracking_end_time_point:
        return None  # Track is not long enough for the requested time window

    # Extract intensites from the track
    intensities = list()
    times_h = list()
    for time_point_number in range(tracking_start_time_point, tracking_end_time_point + 1):
        position = track.find_position_at_time_point_number(time_point_number)
        intensity = position_markers.get_normalized_intensity(experiment, position)
        if intensity is None:
            return None  # We don't have a full data set

        times_h.append(time_point_number * resolution.time_point_interval_h)
        intensities.append(intensity)

    # Calculate statistics
    mean = numpy.mean(intensities)
    mean_stderr = numpy.std(intensities, ddof=1) / math.sqrt(len(intensities))
    slope, _, _, _, slope_stderr = linregress(times_h, intensities)
    return IntensityOverTime(mean=float(mean), mean_stderr=float(mean_stderr),
                             slope=float(slope), slope_stderr=float(slope_stderr))
