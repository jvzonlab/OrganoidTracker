import unittest

from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection


class TestPositionCollection(unittest.TestCase):

    def test_move_time(self):
        positions = PositionCollection()

        positions.add(Position(1, 2, 3, time_point_number=4))

        # Move in time, and check if move was successful
        positions.move_in_time(10)
        self.assertFalse(positions.contains_position(Position(1, 2, 3, time_point_number=4)))
        self.assertTrue(positions.contains_position(Position(1, 2, 3, time_point_number=14)))
