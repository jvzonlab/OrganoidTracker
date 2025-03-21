from typing import Tuple, Union, Optional, Literal

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint


class ImageResolution:
    """Represents the resolution of a 3D image. X and y resolution must be equal. The fields in this class should be
    treated as immutable: don't modify their values after creation."""

    # If you want to define 1 um = 1 px, use this resolution. Useful if you want ot measure distances in pixels instead
    # of micrometers.
    PIXELS: "ImageResolution" = ...  # Initialized after definition of class.

    pixel_size_zyx_um: Tuple[float, float, float]
    time_point_interval_m: float  # Time between time points in minutes

    def __init__(self, pixel_size_x_um: float, pixel_size_y_um: float, pixel_size_z_um: float, time_point_interval_m: float):
        if pixel_size_x_um < 0 or pixel_size_z_um < 0 or time_point_interval_m < 0:
            raise ValueError("Resolution cannot be negative")

        self.pixel_size_zyx_um = (pixel_size_z_um, pixel_size_y_um, pixel_size_x_um)
        self.time_point_interval_m = time_point_interval_m

    @property
    def time_point_interval_h(self) -> float:
        return self.time_point_interval_m / 60

    @property
    def pixel_size_x_um(self) -> float:
        return self.pixel_size_zyx_um[2]

    @property
    def pixel_size_y_um(self) -> float:
        return self.pixel_size_zyx_um[1]

    @property
    def pixel_size_z_um(self) -> float:
        return self.pixel_size_zyx_um[0]

    def __repr__(self) -> str:
        return f"ImageResolution({self.pixel_size_x_um}, {self.pixel_size_y_um}, {self.pixel_size_z_um}," \
               f" {self.time_point_interval_m})"

    def is_incomplete(self, *, require_time_resolution: bool = True, require_z: bool = True) -> bool:
        """Returns True if the x, y, z and/or t resolution is zero. Otherwise it returns False."""
        if require_time_resolution:
            if self.time_point_interval_m == 0 or self.time_point_interval_m == float("inf"):
                return True
        if require_z:
            if self.pixel_size_z_um == 0:
                return True
        return self.pixel_size_x_um == 0 or self.pixel_size_y_um == 0


# See typehint at beginning of ImageResolution class
ImageResolution.PIXELS = ImageResolution(1, 1, 1, 1)


class ImageTimings:
    """For working with experiments where the timing between time points is variable. (Although a constant timing is
    also supported).

    If you request timing information for a time point before the available range, then the time interval between the
    first and second time point is used. If you request timing information for a time point after the available range,
    then the timing between the last two time points is used.

    You need to specify the timing for at least two subsequent time points, of which the first is set at time 0.

    The object is immutable, and it's best to keep it that way. The Images class hands out a fresh timings instance
    based on the current value of ImageResolution.time_point_interval_m if no explicit timings are available. This would
    not be possible if the class were mutable, as then any changes would be expected to be stored in the experiment.
    """

    _min_time_point_number: int
    _timings_m: ndarray

    @staticmethod
    def contant_timing(time_resolution_m: float) -> "ImageTimings":
        """For dealing with a constant time resolution."""
        return ImageTimings(0, numpy.array([0, time_resolution_m], dtype=numpy.float64))

    def __init__(self, min_time_point_number: int, cumulative_timings_m: ndarray):
        """Allows you to specify a variable time resolution."""
        self._min_time_point_number = min_time_point_number
        if len(cumulative_timings_m) < 2:
            raise ValueError("Need at least two time points")
        if cumulative_timings_m.dtype != numpy.float64:
            raise ValueError("Data type of cumulative timings need to be numpy.float64, not " + str(cumulative_timings_m.dtype))
        self._timings_m = cumulative_timings_m

    def get_time_m_since_start(self, time_point: Union[int, TimePoint]) -> float:
        """Gets the amount of time (in minutes) elapsed since the start of the experiment."""
        time_point_number = time_point.time_point_number() if isinstance(time_point, TimePoint) else time_point
        if time_point_number < self._min_time_point_number:
            # For example requesting the time of time point number -3, when the minimum is 0
            interval_m = self._timings_m[1] - self._timings_m[0]
            steps_before_min = self._min_time_point_number - time_point_number
            return self._timings_m[0] - steps_before_min * interval_m
        if time_point_number >= self._min_time_point_number + len(self._timings_m):
            # Requesting the time after the last time point that we have timings for
            interval_m = self._timings_m[-1] - self._timings_m[-2]
            steps_after_max = time_point_number - self._min_time_point_number - len(self._timings_m) + 1
            return self._timings_m[-1] + steps_after_max * interval_m
        # In range, easy
        return self._timings_m[time_point_number - self._min_time_point_number]

    def get_time_h_since_start(self, time_point: Union[int, TimePoint]) -> float:
        """Gets the amount of time (in hours) elapsed since the start of the experiment."""
        return self.get_time_m_since_start(time_point) / 60

    def get_time_m_since_previous(self, time_point: Union[int, TimePoint]) -> float:
        """Gets the amount of time (in minutes) elapsed since the previous time point."""
        return self.get_time_m_since_start(time_point) - self.get_time_m_since_start(time_point - 1)

    def min_time_point_number(self) -> int:
        """Gets the lowest time point number for which we have the timings. The time will always be zero at this point.
        """
        return self._min_time_point_number

    def max_time_point_number(self) -> int:
        """Gets the highest time point number for which we have the timings."""
        return self._min_time_point_number + len(self._timings_m) - 1

    def limit_to_time(self, min_time_point_number: int, max_time_point_number: int) -> Optional["ImageTimings"]:
        """Returns a new instance with all timing information for time points outside the given range deleted."""
        new_min = max(self.min_time_point_number(), min_time_point_number)
        new_max = min(self.max_time_point_number(), max_time_point_number)

        offset = new_min - self._min_time_point_number
        if offset > len(self._timings_m) - 2:
            return None  # Not enough data points remain
        count = new_max - new_min + 1
        if offset + count > len(self._timings_m):
            count = len(self._timings_m) - offset
            if count < 2:
                return None  # Not enough data points remain

        return ImageTimings(new_min, self._timings_m[offset:offset+count].copy())

    def is_simple_multiplication(self) -> bool:
        """Checks if the timings object is just calculating time_point * dt, or whether more complex timings are stored.
        """
        if abs(self.get_time_m_since_start(0)) > 0.0000001:
            return False  # Doesn't start at 0, so cannot be a simple multiplication

        # If only one difference is specified, then for sure they're all equal
        if len(self._timings_m) == 2:
            return True

        # Check if all differences are approximately equal
        delta = self._timings_m[1:] - self._timings_m[:-1]
        return numpy.allclose(delta - delta[0], 0)

    def get_cumulative_timings_array_m(self) -> ndarray:
        """Gets a copy of the cumulative timings array. Pay attention to the fact that position 0 in the array
        corresponds to self.min_time_point_number(), and not necessarily to 0.

        Normally, you don't need this method, it's mostly for serialization purposes."""
        return self._timings_m.copy()

    def copy(self) -> "ImageTimings":
        """Returns a copy of the current object. Changes to the copy won't write through to this object."""
        return ImageTimings(self._min_time_point_number, self._timings_m.copy())

    def move_in_time(self, time_point_delta: int):
        """Moves the timings the given amount of time points in time. So if delta is 2, then the reported timings of
        time point 2 will now be what the reported timings of time point 0 were."""
        self._min_time_point_number += time_point_delta

    def find_closest_time_point(self, time: float, unit: Union[Literal["m"], Literal["h"]]) -> TimePoint:
        """Finds the time point closest to the given time in minutes or hours. If two time points are equally close,
        then the later one is returned."""
        if unit not in ("m", "h"):
            raise ValueError(f"Unit must be 'm' or 'h', was '{unit}'")
        time_m = time if unit == "m" else time * 60

        if time_m < self._timings_m[0]:
            # Calculate how many time points back we need to go
            time_interval_m = self._timings_m[1] - self._timings_m[0]
            time_before_array_m = time_m - self._timings_m[0]
            return TimePoint(self._min_time_point_number + _round(time_before_array_m / time_interval_m))

        if time_m >= self._timings_m[-1]:
            # Calculate how many time points forward we need to go
            time_interval_m = self._timings_m[-1] - self._timings_m[-2]
            time_after_array_m = time_m - self._timings_m[-1]
            return TimePoint(self._min_time_point_number + len(self._timings_m) - 1 + _round(time_after_array_m / time_interval_m))

        # In range
        time_index = numpy.searchsorted(self._timings_m, time_m, side="right")
        difference_to_previous_time_point = time_m - self._timings_m[time_index - 1]
        difference_to_next_time_point = self._timings_m[time_index] - time_m
        if difference_to_next_time_point <= difference_to_previous_time_point:
            return TimePoint(self._min_time_point_number + time_index)
        else:
            return TimePoint(self._min_time_point_number + time_index - 1)


def _round(number: float) -> int:
    """In Python, 0.5 rounds to 0, as opposed to 1, as any halves are rounded to the nearest even number.
    This function rounds 0.5 to 1. Also for negative numbers it rounds UP for halves, so -0.5 rounds to 0."""
    if number < 0:
        # Handle negative numbers
        absolute_number = abs(number)
        remainder = absolute_number % 1
        if remainder > 0.5:
            # Note that int(-0.8) is 0, so we need to subtract 1 from the result to get -1 in this example
            return int(number) - 1
        else:
            return int(number)

    remainder = number % 1
    if remainder < 0.5:
        return int(number)
    return int(number) + 1
