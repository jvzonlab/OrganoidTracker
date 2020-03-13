import unittest

from organoid_tracker.core.warning_limits import WarningLimits


class TestWarningLimits(unittest.TestCase):

    def test_to_and_from_dict(self):
        warning_limits = WarningLimits()
        warning_limits.max_distance_moved_um_per_min = 60

        copy = WarningLimits(**warning_limits.to_dict())
        self.assertEqual(warning_limits.max_distance_moved_um_per_min, copy.max_distance_moved_um_per_min)
