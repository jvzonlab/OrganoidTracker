"""Warnings limits: if these limits are reached, the error checker will raise a warning."""
from typing import Dict, Optional, Set, List

from organoid_tracker.core import TimePoint


class WarningLimits:
    """Mutable object; states all the limits that will trigger warnings if they are violated."""

    # Minimum of time in between divisions in hours.
    min_time_between_divisions_h: float

    # Maximum distance a cell can move in micrometers/minute.
    max_distance_moved_um_per_min: float

    # Minimum probability of a link/division
    min_probability: float

    # Minimum probability of a marginal, should be a lot higher than the normal minimum probability
    # (If a link has a probability of 0.2, but there are no other options, it will still have a high marginal
    # probability. So the marginal probability cutoff can be a lot higher.)
    min_marginal_probability: float

    # Minimum time point at which to start error correction
    min_time_point: Optional[TimePoint] = None

    # Last time point in which errors are warned for
    max_time_point: Optional[TimePoint] = None

    # Error type ids that we do not want to warn for
    excluded_errors: Set[int]

    def __init__(self, *, min_time_between_divisions_h: float = 10,
                 max_distance_moved_um_per_min: float = 10/12,
                 min_probability: float = 0.1,
                 min_marginal_probability: float = 0.99,
                 min_time_point_number: Optional[int] = None,
                 max_time_point_number: Optional[int] = None,
                 excluded_errors: Optional[List[int]] = None,
                 **kwargs):
        """Initializes the warning limits. Normally, this method will be called with data coming from a JSON dictionary.

        Note: any data stored in the kwargs parameter is discarded. This parameter exists to make the program
        forwards compatible. If a future version of OrganoidTracker introduces a new warning limit, then this version
        will simply load (and discard) that warning limit, instead of crashing.
        """
        self.min_time_between_divisions_h = min_time_between_divisions_h
        self.max_distance_moved_um_per_min = max_distance_moved_um_per_min
        self.min_probability = min_probability
        self.min_marginal_probability = min_marginal_probability
        self.min_time_point = TimePoint(min_time_point_number) \
            if min_time_point_number is not None else None
        self.max_time_point = TimePoint(max_time_point_number) \
            if max_time_point_number is not None else None
        self.excluded_errors = set(excluded_errors) if excluded_errors is not None else set()

    def to_dict(self) -> Dict[str, float]:
        """Gets all settings in a dictionary ready for JSON serialization. `WarningLimits(**limits.to_dict())` will
        result in an object with the same values."""
        return {
            "min_time_between_divisions_h": self.min_time_between_divisions_h,
            "max_distance_moved_um_per_min": self.max_distance_moved_um_per_min,
            "min_probability": self.min_probability,
            "min_marginal_probability": self.min_marginal_probability,
            "min_time_point_number": self.min_time_point.time_point_number()
                if self.min_time_point is not None else None,
            "max_time_point_number": self.max_time_point.time_point_number()
                if self.max_time_point is not None else None,
            "excluded_errors": list(self.excluded_errors)
        }
