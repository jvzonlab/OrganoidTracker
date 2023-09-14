import unittest

import numpy

from organoid_tracker.core import TimePoint
from organoid_tracker.core.resolution import ImageTimings


class TestWarningLimits(unittest.TestCase):

    def test_constant_timings(self):
        timings = ImageTimings.contant_timing(10)
        self.assertEqual(-20, timings.get_time_m_since_start(TimePoint(-2)))
        self.assertEqual(-10, timings.get_time_m_since_start(TimePoint(-1)))
        self.assertEqual(0, timings.get_time_m_since_start(TimePoint(0)))
        self.assertEqual(10, timings.get_time_m_since_start(TimePoint(1)))
        self.assertEqual(20, timings.get_time_m_since_start(TimePoint(2)))
        self.assertEqual(30, timings.get_time_m_since_start(TimePoint(3)))

    def test_variable_timings(self):
        # Here, we start at a one-minute timing, then switch to 5 minute intervals
        # Time point 1 is defined at zero minutes
        timings = ImageTimings(1, numpy.array([0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30], dtype=numpy.float64))
        self.assertEqual(-2, timings.get_time_m_since_start(TimePoint(-1)))
        self.assertEqual(-1, timings.get_time_m_since_start(TimePoint(0)))
        self.assertEqual(0, timings.get_time_m_since_start(TimePoint(1)))
        self.assertEqual(1, timings.get_time_m_since_start(TimePoint(2)))
        self.assertEqual(2, timings.get_time_m_since_start(TimePoint(3)))
        self.assertEqual(3, timings.get_time_m_since_start(TimePoint(4)))
        self.assertEqual(4, timings.get_time_m_since_start(TimePoint(5)))
        self.assertEqual(5, timings.get_time_m_since_start(TimePoint(6)))
        self.assertEqual(10, timings.get_time_m_since_start(TimePoint(7)))
        self.assertEqual(15, timings.get_time_m_since_start(TimePoint(8)))
        self.assertEqual(20, timings.get_time_m_since_start(TimePoint(9)))
        self.assertEqual(25, timings.get_time_m_since_start(TimePoint(10)))
        self.assertEqual(30, timings.get_time_m_since_start(TimePoint(11)))
        self.assertEqual(35, timings.get_time_m_since_start(TimePoint(12)))
        self.assertEqual(40, timings.get_time_m_since_start(TimePoint(13)))

    def test_raw_int(self):
        # Can we also use int, instead of TimePoint?
        timings = ImageTimings.contant_timing(10)
        self.assertEqual(-20, timings.get_time_m_since_start(-2))

    def test_time_limit(self):
        timings = ImageTimings(1, numpy.array([0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30], dtype=numpy.float64))

        # Remove the 5-minute intervals
        timings_cut = timings.limit_to_time(min_time_point_number=2, max_time_point_number=6)

        # Check if five-minute intervals are no longer recognized
        self.assertEqual(10, timings.get_time_m_since_start(TimePoint(7)))
        self.assertEqual(6, timings_cut.get_time_m_since_start(TimePoint(7)))

        # Check if time point 1 still calculates as zero
        self.assertEqual(0, timings_cut.get_time_m_since_start(TimePoint(1)))

    def test_simple_multiplication(self):
        self.assertTrue(ImageTimings(0, numpy.array([0, 1, 2, 3, 4, 5],
                                                    dtype=numpy.float64)).is_simple_multiplication())
        self.assertFalse(ImageTimings(1, numpy.array([0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
                                                     dtype=numpy.float64)).is_simple_multiplication())

        self.assertTrue(ImageTimings(1, numpy.array([1, 2, 3, 4, 5],
                                                    dtype=numpy.float64)).is_simple_multiplication())
        self.assertFalse(ImageTimings(2, numpy.array([1, 2, 3, 4, 5],
                                                     dtype=numpy.float64)).is_simple_multiplication())
        self.assertTrue(ImageTimings.contant_timing(10).is_simple_multiplication())