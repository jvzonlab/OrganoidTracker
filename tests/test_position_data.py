import unittest

from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData


class TestPositionData(unittest.TestCase):

    def test_data(self):
        position = Position(0, 0, 0, time_point_number=0)
        position_data = PositionData()
        position_data.set_position_data(position, "name", "AA")

        self.assertEqual("AA", position_data.get_position_data(position, "name"))

    def test_has_position_data(self):
        position_data = PositionData()
        position = Position(3, 5, 6, time_point_number=5)
        self.assertFalse(position_data.has_position_data_with_name("test_data"))
        position_data.set_position_data(position, "test_data", True)
        self.assertTrue(position_data.has_position_data_with_name("test_data"))
        position_data.set_position_data(position, "test_data", None)
        self.assertFalse(position_data.has_position_data_with_name("test_data"))
