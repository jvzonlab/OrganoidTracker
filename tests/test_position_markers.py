import unittest

from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.position_analysis import position_markers


class TestPositionData(unittest.TestCase):


    def test_normalization(self):
        position_data = PositionData()
        position_data.set_position_data(Position(0, 0, 0, time_point_number=0), "intensity", 8)
        position_data.set_position_data(Position(1, 0, 0, time_point_number=0), "intensity", 10)
        position_data.set_position_data(Position(2, 0, 0, time_point_number=0), "intensity", 12)

        normalizer = position_markers.get_intensity_normalization(position_data)
        self.assertEqual(50, normalizer.factor)
        self.assertEqual(-400, normalizer.offset)

