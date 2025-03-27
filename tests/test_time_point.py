import unittest

from organoid_tracker.core import TimePoint


class TestTimePoint(unittest.TestCase):

    def test_basics(self):
        self.assertEqual(20, TimePoint(20).time_point_number())
        self.assertEqual(-20, TimePoint(-20).time_point_number())

    def test_comparison(self):
        self.assertTrue(TimePoint(20) == TimePoint(20))
        self.assertTrue(TimePoint(20) != TimePoint(21))
        self.assertTrue(TimePoint(20) < TimePoint(21))
        self.assertTrue(TimePoint(20) <= TimePoint(21))
        self.assertTrue(TimePoint(21) > TimePoint(20))
        self.assertTrue(TimePoint(21) >= TimePoint(20))

    def test_iterator(self):
        time_point_range = TimePoint.range(TimePoint(5), TimePoint(7))
        self.assertEqual(3, len(time_point_range))

        self.assertEqual(TimePoint(5), time_point_range[0])
        self.assertEqual(TimePoint(6), time_point_range[1])
        self.assertEqual(TimePoint(7), time_point_range[2])

        self.assertEqual([TimePoint(5), TimePoint(6), TimePoint(7)], list(time_point_range))

    def test_one_time_point_iterator(self):
        time_point_range = TimePoint.range(TimePoint(5), TimePoint(5))
        self.assertEqual(1, len(time_point_range))
        self.assertEqual([TimePoint(5)], list(time_point_range))

    def test_none_iterator(self):
        time_point_range = TimePoint.range(None, None)
        self.assertEqual(0, len(time_point_range))
        self.assertEqual([], list(time_point_range))

