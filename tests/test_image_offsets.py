import unittest

from ai_track.core import TimePoint
from ai_track.core.images import ImageOffsets
from ai_track.core.position import Position


class TestImageOffsets(unittest.TestCase):

    def test_basics(self):
        offsets = ImageOffsets()
        self.assertEquals(Position(0, 0, 0), offsets.of_time_point(TimePoint(15)))

        offsets.update_offset(10, 9, 8, min_time_point=12, max_time_point=18)
        self.assertEquals(Position(0, 0, 0), offsets.of_time_point(TimePoint(11)))  # Just outside range
        self.assertEquals(Position(10, 9, 8), offsets.of_time_point(TimePoint(12)))  # Just inside
        self.assertEquals(Position(10, 9, 8), offsets.of_time_point(TimePoint(13)))  # Inside
        self.assertEquals(Position(10, 9, 8), offsets.of_time_point(TimePoint(18)))  # Just inside
        self.assertEquals(Position(0, 0, 0), offsets.of_time_point(TimePoint(19)))  # Just outside

    def test_negative_time_point(self):
        offsets = ImageOffsets()
        self.assertEquals(Position(0, 0, 0), offsets.of_time_point(TimePoint(-15)))

        offsets.update_offset(10, 9, 8, min_time_point=-18, max_time_point=-12)
        self.assertEquals(Position(0, 0, 0), offsets.of_time_point(TimePoint(-11)))  # Just outside range
        self.assertEquals(Position(10, 9, 8), offsets.of_time_point(TimePoint(-12)))  # Just inside
        self.assertEquals(Position(10, 9, 8), offsets.of_time_point(TimePoint(-13)))  # Inside
        self.assertEquals(Position(10, 9, 8), offsets.of_time_point(TimePoint(-18)))  # Just inside
        self.assertEquals(Position(0, 0, 0), offsets.of_time_point(TimePoint(-19)))  # Just outside
