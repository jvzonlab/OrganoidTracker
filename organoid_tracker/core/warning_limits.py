"""Warnings limits: if these limits are reached, the error checker will raise a warning."""
from typing import Dict


class WarningLimits:
    """Mutable object; states all the limits that will trigger warnings if they are violated."""

    # Minimum of time in between divisions in hours.
    min_time_between_divisions_h: float

    # Maximum distance a cell can move in micrometers/minute.
    max_distance_moved_um_per_min: float

    # Minimum probability of a link/division
    min_probability: float

    def __init__(self, *, min_time_between_divisions_h: float = 10,
                 max_distance_moved_um_per_min: float = 10/12,
                 min_probability: float = 0.1):
        self.min_time_between_divisions_h = min_time_between_divisions_h
        self.max_distance_moved_um_per_min = max_distance_moved_um_per_min
        self.min_probability = min_probability

    def to_dict(self) -> Dict[str, float]:
        """Gets all settings in a dictionary. `WarningLimits(**limits.to_dict())` will result in an object with the same
        values."""
        return {
            "min_time_between_divisions_h": self.min_time_between_divisions_h,
            "max_distance_moved_um_per_min": self.max_distance_moved_um_per_min
        }
