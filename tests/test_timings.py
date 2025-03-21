import unittest

import numpy

from organoid_tracker.core import TimePoint
from organoid_tracker.core.resolution import ImageTimings


class TestTimings(unittest.TestCase):

    def test_basics(self):
        # Test with four time points, jump in time between time point 2 and 3, otherwise 10 minute intervals
        timings = ImageTimings(0, numpy.array([0, 10, 90, 100], dtype=numpy.float64))

        self.assertEqual(0, timings.min_time_point_number())
        self.assertEqual(3, timings.max_time_point_number())

        # Checks in range
        self.assertEqual(0, timings.get_time_m_since_start(0))
        self.assertEqual(10, timings.get_time_m_since_start(1))
        self.assertEqual(90, timings.get_time_m_since_start(2))
        self.assertEqual(100, timings.get_time_m_since_start(3))

        # Checks out of range (will use extrapolation)
        self.assertEqual(-10, timings.get_time_m_since_start(-1))
        self.assertEqual(-20, timings.get_time_m_since_start(-2))
        self.assertEqual(110, timings.get_time_m_since_start(4))
        self.assertEqual(120, timings.get_time_m_since_start(5))

    def test_inverse_search(self):
        # Test with four time points, jump in time between time point 2 and 3, otherwise 10 minute intervals
        timings = ImageTimings(0, numpy.array([0, 10, 90, 100], dtype=numpy.float64))

        # Test jump from first to second time point
        self.assertEqual(TimePoint(0), timings.find_closest_time_point(0, "m"))
        self.assertEqual(TimePoint(0), timings.find_closest_time_point(4, "m"))
        self.assertEqual(TimePoint(1), timings.find_closest_time_point(5, "m"))
        self.assertEqual(TimePoint(1), timings.find_closest_time_point(10, "m"))

        # Test jump from second to third time point
        self.assertEqual(TimePoint(1), timings.find_closest_time_point(49, "m"))
        self.assertEqual(TimePoint(2), timings.find_closest_time_point(50, "m"))

        # Test time points after the last time point
        self.assertEqual(TimePoint(3), timings.find_closest_time_point(104, "m"))
        self.assertEqual(TimePoint(4), timings.find_closest_time_point(105, "m"))
        self.assertEqual(TimePoint(4), timings.find_closest_time_point(110, "m"))

        # Test time points before the first time point
        self.assertEqual(TimePoint(0), timings.find_closest_time_point(-1, "m"))
        self.assertEqual(TimePoint(0), timings.find_closest_time_point(-4, "m"))
        # In case of equally close by time points, we round to the NEXT time point
        self.assertEqual(TimePoint(0), timings.find_closest_time_point(-5, "m"))
        self.assertEqual(TimePoint(-1), timings.find_closest_time_point(-6, "m"))
        self.assertEqual(TimePoint(-1), timings.find_closest_time_point(-10, "m"))

        # Test using hours as unit
        self.assertEqual(TimePoint(1), timings.find_closest_time_point(1, "h"))
