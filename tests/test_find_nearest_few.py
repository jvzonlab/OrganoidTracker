import unittest

from autotrack.core import TimePoint
from autotrack.core.positions import Position
from autotrack.linking.nearby_position_finder import find_close_positions


class TestFindNearestFew(unittest.TestCase):

    def test_find_two(self):
        time_point = TimePoint(2)
        positions = set()
        positions.add(Position(10,20,0, time_point=time_point))
        positions.add(Position(11,20,0, time_point=time_point))
        positions.add(Position(100,20,0, time_point=time_point))
        found = find_close_positions(positions, Position(40, 20, 0), 1.1)
        self.assertEqual(2, len(found), "Expected to find two positions that are close to each other")

    def test_find_one(self):
        time_point = TimePoint(2)
        positions = set()
        positions.add(Position(10, 20, 0, time_point=time_point))
        positions.add(Position(11, 20, 0, time_point=time_point))
        positions.add(Position(100, 20, 0, time_point=time_point))
        found = find_close_positions(positions, Position(80, 20, 0), 1.1)
        self.assertEqual(1, len(found), "Expected to find one position that is close enough")

    def test_zero_tolerance(self):
        time_point = TimePoint(2)
        positions = set()
        positions.add(Position(10, 20, 0, time_point=time_point))
        positions.add(Position(11, 20, 0, time_point=time_point))
        positions.add(Position(100, 20, 0, time_point=time_point))
        found = find_close_positions(positions, Position(40, 20, 0), 1)
        self.assertEqual(1, len(found), "Tolerance is set to 1.0, so only one position may be found")
