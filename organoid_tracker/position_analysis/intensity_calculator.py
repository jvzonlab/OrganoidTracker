"""Contains a lot of functions related to measuring intensity, averaged intensity and intensity derivatives."""

import math
from typing import Optional, Dict, List

import numpy
from scipy.stats import linregress

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection

# The default intensity key, used if the user didn't specify another one
DEFAULT_INTENSITY_KEY = "intensity"

class IntensityOverTime:
    """Represents an intensity over time. The object holds the raw values, and statistics are calculated dynamically."""
    _times_h: List[float]
    _intensities: List[float]

    _mean: Optional[float] = None
    _mean_stderr: Optional[float] = None
    _slope: Optional[float] = None
    _slope_stderr: Optional[float] = None

    def __init__(self, times_h: List[float], intensities: List[float]):
        self._times_h = times_h
        self._intensities = intensities

    def multiply(self, factor: float):
        """Multiplies all intensities with a factor."""
        # First reset the statistics
        self._mean = None
        self._mean_stderr = None
        self._slope = None
        self._slope_stderr = None
        # Then multiply the intensities
        for i in range(len(self._intensities)):
            self._intensities[i] = self._intensities[i] * factor

    @property
    def mean(self) -> float:
        if self._mean is None:
            self._mean = sum(self._intensities) / len(self._intensities)
        return self._mean

    @property
    def mean_stderr(self) -> float:
        if self._mean_stderr is None:
            if len(self._intensities) > 1:
                self._mean_stderr = numpy.std(self._intensities, ddof=1) / math.sqrt(len(self._intensities))
            else:
                self._mean_stderr = 0
        return self._mean_stderr

    @property
    def slope(self) -> float:
        if self._slope is None:
            if len(self._intensities) > 1:
                self._slope, _, _, _, self._slope_stderr = linregress(self._times_h, self._intensities)
            else:
                self._slope = 0
        return self._slope

    @property
    def slope_stderr(self) -> float:
        if self._slope_stderr is None:
            if len(self._intensities) > 1:
                self._slope, _, _, _, self._slope_stderr = linregress(self._times_h, self._intensities)
            else:
                self._slope_stderr = 0
        return self._slope_stderr

    def get_production(self, degradation_rate_h: float) -> float:
        """Calculates the production using a simple production-degradation model
            f
        ∅  ⇄  G
            b
        """
        if self.mean <= 0:
            return 0  # Assume the actual slope and mean are zero
        return degradation_rate_h * self.mean + self.slope


def get_normalized_intensity_over_time(experiment: Experiment, around_position: Position, time_window_h: float, *,
                                       allow_incomplete: bool = False, intensity_key: str = DEFAULT_INTENSITY_KEY,
                                       per_pixel: bool = False) -> Optional[IntensityOverTime]:
    """Gets the slope and mean intensity over time given time span. Returns None if not enough data is
    available, which is the case if the track is too short or if some intensities are missing."""
    resolution = experiment.images.resolution()
    time_window_tp = int(math.ceil(time_window_h / resolution.time_point_interval_h))
    tracking_start_time_point = int(around_position.time_point_number() - time_window_tp / 2)
    tracking_end_time_point = int(math.ceil(around_position.time_point_number() + time_window_tp / 2))

    # Find the track
    track = experiment.links.get_track(around_position)
    if track is None:
        return None
    if track.first_time_point_number() > tracking_start_time_point:
        if not allow_incomplete:
            return None  # Track is not long enough for the requested time window
        tracking_start_time_point = track.first_time_point_number()
    if track.last_time_point_number() < tracking_end_time_point:
        if not allow_incomplete:
            return None
        tracking_end_time_point = track.last_time_point_number()

    # Extract intensites from the track
    intensities = list()
    times_h = list()
    for time_point_number in range(tracking_start_time_point, tracking_end_time_point + 1):
        position = track.find_position_at_time_point_number(time_point_number)
        intensity = get_normalized_intensity(experiment, position, intensity_key=intensity_key, per_pixel=per_pixel)
        if intensity is None:
            # We don't have a full data set
            if not allow_incomplete:
                return None
            continue

        times_h.append(time_point_number * resolution.time_point_interval_h)
        intensities.append(intensity)

    if len(intensities) == 0:
        return None  # Will cause problems

    return IntensityOverTime(times_h, intensities)


def set_raw_intensities(experiment: Experiment, raw_intensities: Dict[Position, float], volumes: Dict[Position, int],
                        *, intensity_key: str = DEFAULT_INTENSITY_KEY):
    """Registers the given intensities for the given positions. Both dicts must have the same keys.

    Will overwrite any previous intensities saved under the given key.

    Will also add this intensity to the intensity_keys of the experiment.

    Also removes any previously set intensity normalization for that key."""
    if raw_intensities.keys() != volumes.keys():
        raise ValueError("Need to supply intensities and volumes for the same cells")
    if len(intensity_key) == 0:
        raise ValueError("Cannot use an empty intensity_key")

    remove_intensities(experiment, intensity_key=intensity_key)
    experiment.positions.add_positions_data(intensity_key, raw_intensities)
    experiment.positions.add_positions_data(intensity_key + "_volume", volumes)


def remove_intensities(experiment: Experiment, *, intensity_key: str = DEFAULT_INTENSITY_KEY):
    """Deletes the intensities with the given key."""

    # Remove values
    experiment.positions.delete_data_with_name(intensity_key)
    experiment.positions.delete_data_with_name(intensity_key + "_volume")

    # Remove normalization
    remove_intensity_normalization(experiment, intensity_key=intensity_key)


def get_intensity_keys(experiment: Experiment) -> List[str]:
    """Gets the keys of all stored intensities.

    Any key (for example "intensity") that is numeric and also has a "_volume" counterpart (like "intensity_volume") is
    seen as being an intensity.
    """
    return_list = list()
    names_and_types = experiment.positions.get_data_names_and_types()
    for data_name, data_type in names_and_types.items():
        if data_type != float:
            continue  # Skip non-numeric metadata

        if data_name.endswith("_volume") and data_name[:-len("_volume")] in names_and_types:
            continue  # Skip volume of an intensity

        if not data_name + "_volume" in names_and_types.keys():
            continue  # Has no measured volume, so cannot be used as an intensity

        return_list.append(data_name)
    return return_list


def get_raw_intensity(positions: PositionCollection, position: Position, *, intensity_key: str = DEFAULT_INTENSITY_KEY
                      ) -> Optional[float]:
    """Gets the raw intensity of the position."""
    return positions.get_position_data(position, intensity_key)


def get_normalized_intensity(experiment: Experiment, position: Position, *, intensity_key: str = DEFAULT_INTENSITY_KEY,
                             per_pixel: bool = False) -> Optional[float]:
    """Gets the normalized intensity of the position. Takes into account the background and the intensity multiplier
    (for normalization), which might be specific to the time point or Z layer. Either returns the intensity sum or the
    intensity per pixel, depending on the per_pixel parameter (default false).
    """
    positions = experiment.positions
    global_data = experiment.global_data

    intensity = positions.get_position_data(position, intensity_key)
    if intensity is None:
        return None
    background_per_px = global_data.get_data(intensity_key + "_background_per_pixel")
    if background_per_px is None:
        background_per_px = 0
    multiplier = global_data.get_data(intensity_key + "_multiplier_z" + str(round(position.z)))
    if multiplier is None:
        # Try time multiplier
        multiplier = global_data.get_data(intensity_key + "_multiplier_t" + str(position.time_point_number()))

        if multiplier is None:
            # Try global multiplier
            multiplier = global_data.get_data(intensity_key + "_multiplier")
            if multiplier is None:
                # Also failed - then don't multiply
                multiplier = 1
    volume_px = positions.get_position_data(position, intensity_key + "_volume")
    if volume_px is None:
        if per_pixel:
            return None  # Can't calculate intensity per pixel if volume is missing

        # Can't do background subtraction if volume is missing
        background_per_px = 0

        # Assume a default volume (volume won't be used, since background is set to 0, and per-pixel intensity
        # is disallowed)
        volume_px = 1

    if per_pixel:
        return (intensity / volume_px - background_per_px) * multiplier
    return (intensity - background_per_px * volume_px) * multiplier


def perform_intensity_normalization(experiment: Experiment, *, background_correction: bool = True, z_correction: bool = False,
                                    time_correction: bool = False, intensity_key: str = DEFAULT_INTENSITY_KEY):
    """Gets the average intensity of all positions in the experiment.
    Returns None if there are no intensity recorded."""
    if time_correction and z_correction:
        raise UserError("Time and Z correction", "Cannot apply both a time and a z correction.")
    remove_intensity_normalization(experiment)

    intensities = list()
    volumes = list()
    zs = list()
    ts = list()

    positions = experiment.positions
    for position, intensity in positions.find_all_positions_with_data(intensity_key):
        volume = positions.get_position_data(position, intensity_key + "_volume")
        if volume is None and background_correction:
            continue
        if intensity == 0:
            continue

        intensities.append(intensity)
        volumes.append(volume)
        zs.append(round(position.z))
        ts.append(position.time_point_number())

    if len(intensities) == 0:
        return

    intensities = numpy.array(intensities, dtype=numpy.float32)
    volumes = numpy.array(volumes, dtype=numpy.float32)
    zs = numpy.array(zs, dtype=numpy.int32)
    ts = numpy.array(ts, dtype=numpy.int32)

    if background_correction:
        # Assume the lowest signal consists of only background
        lowest_intensity_index = numpy.argmin(intensities / volumes)
        background_per_px = float(intensities[lowest_intensity_index] / volumes[lowest_intensity_index])

        # Subtract this background
        intensities -= volumes * background_per_px
        experiment.global_data.set_data(intensity_key + "_background_per_pixel", background_per_px)
    else:
        experiment.global_data.set_data(intensity_key + "_background_per_pixel", 0)

    # Now normalize the mean to 1
    if z_correction:
        for z in range(int(numpy.min(zs)), int(numpy.max(zs)) + 1):
            median = numpy.median(intensities[zs == z])
            normalization_factor = float(1 / median)
            experiment.global_data.set_data(intensity_key + "_multiplier_z" + str(z), normalization_factor)
    elif time_correction:
        for t in range(int(numpy.min(ts)), int(numpy.max(ts)) + 1):
            median = numpy.median(intensities[ts == t])
            normalization_factor = float(1 / median)
            experiment.global_data.set_data(intensity_key + "_multiplier_t" + str(t), normalization_factor)
    else:
        median = numpy.median(intensities)
        normalization_factor = float(1 / median)
        experiment.global_data.set_data(intensity_key + "_multiplier", normalization_factor)


def remove_intensity_normalization(experiment: Experiment, *, intensity_key: str = DEFAULT_INTENSITY_KEY):
    """Removes the normalization set by perform_intensity_normalization."""
    experiment.global_data.set_data(intensity_key + "_background_per_pixel", None)
    experiment.global_data.set_data(intensity_key + "_multiplier", None)
    for key in list(experiment.global_data.get_all_data().keys()):
        if key.startswith(intensity_key + "_multiplier_z") or key.startswith(intensity_key + "_multiplier_t"):
            experiment.global_data.set_data(key, None)

