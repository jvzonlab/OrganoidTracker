import unittest

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.linking.nearby_position_finder import find_close_positions


_PX_RESOLUTION = ImageResolution(1, 1, 1, 1)


class TestFindNearestFew(unittest.TestCase):

    def test_find_two(self):
        time_point = TimePoint(2)
        positions = set()
        positions.add(Position(10,20,0, time_point=time_point))
        positions.add(Position(11,20,0, time_point=time_point))
        positions.add(Position(100,20,0, time_point=time_point))
        found = find_close_positions(positions, around=Position(40, 20, 0), tolerance=1.1, resolution=_PX_RESOLUTION)
        self.assertEqual(2, len(found), "Expected to find two positions that are close to each other")

    def test_find_one(self):
        time_point = TimePoint(2)
        positions = set()
        positions.add(Position(10, 20, 0, time_point=time_point))
        positions.add(Position(11, 20, 0, time_point=time_point))
        positions.add(Position(100, 20, 0, time_point=time_point))
        found = find_close_positions(positions, around=Position(80, 20, 0), tolerance=1.1, resolution=_PX_RESOLUTION)
        self.assertEqual(1, len(found), "Expected to find one position that is close enough")

    def test_zero_tolerance(self):
        time_point = TimePoint(2)
        positions = set()
        positions.add(Position(10, 20, 0, time_point=time_point))
        positions.add(Position(11, 20, 0, time_point=time_point))
        positions.add(Position(100, 20, 0, time_point=time_point))
        found = find_close_positions(positions, around=Position(40, 20, 0), tolerance=1, resolution=_PX_RESOLUTION)
        self.assertEqual(1, len(found), "Tolerance is set to 1.0, so only one position may be found")
