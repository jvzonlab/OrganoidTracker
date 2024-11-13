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

    def test_time_offset(self):
        position_data = PositionData()
        position_data.set_position_data(Position(3, 5, 6, time_point_number=5), "test_data", True)

        position_data.move_in_time(3)  # Move by 3, check if data can now be found in the right slot
        self.assertIsNone(position_data.get_position_data(Position(3, 5, 6, time_point_number=5), "test_data"))
        self.assertTrue(position_data.get_position_data(Position(3, 5, 6, time_point_number=8), "test_data"))

    def test_metadata_merge(self):
        main_pos = Position(3, 5, 6, time_point_number=5)
        other_pos = Position(0, 0, 0, time_point_number=5)

        position_data_a = PositionData()
        position_data_a.set_position_data(main_pos, "test_data_1", "foo")
        position_data_a.set_position_data(main_pos, "test_data_2", "bar")

        # Of b, we overwrite test_data_1, and add test_data_3, and also add test_data_1 for a different position
        position_data_b = PositionData()
        position_data_b.set_position_data(main_pos, "test_data_1", "overwriting")
        position_data_b.set_position_data(main_pos, "test_data_3", "baz")
        position_data_b.set_position_data(other_pos, "test_data_1", "bam")

        # Now merge them, and check if the data is as expected
        position_data_a.merge_data(position_data_b)
        self.assertEqual("overwriting", position_data_a.get_position_data(main_pos, "test_data_1"))
        self.assertEqual("bar", position_data_a.get_position_data(main_pos, "test_data_2"))
        self.assertEqual("baz", position_data_a.get_position_data(main_pos, "test_data_3"))
        self.assertEqual("bam", position_data_a.get_position_data(other_pos, "test_data_1"))

        self.assertEqual({"test_data_1": str, "test_data_2": str, "test_data_3": str},
                         position_data_a.get_data_names_and_types())
