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

    def test_data(self):
        position = Position(0, 0, 0, time_point_number=0)
        positions = PositionCollection()
        positions.add(position)
        positions.set_position_data(position, "name", "AA")

        self.assertEqual("AA", positions.get_position_data(position, "name"))

    def test_has_position_data(self):
        positions = PositionCollection()
        position = Position(3, 5, 6, time_point_number=5)
        positions.add(position)

        self.assertFalse(positions.has_position_data_with_name("test_data"))
        positions.set_position_data(position, "test_data", True)
        self.assertTrue(positions.has_position_data_with_name("test_data"))
        positions.set_position_data(position, "test_data", None)
        self.assertFalse(positions.has_position_data_with_name("test_data"))

    def test_time_offset(self):
        positions = PositionCollection()
        old_pos = Position(3, 5, 6, time_point_number=5)
        positions.add(old_pos)
        positions.set_position_data(old_pos, "test_data", True)

        positions.move_in_time(3)  # Move by 3, check if data can now be found in the right slot
        self.assertIsNone(positions.get_position_data(old_pos, "test_data"))
        self.assertTrue(positions.get_position_data(Position(3, 5, 6, time_point_number=8), "test_data"))

    def test_metadata_merge(self):
        main_pos = Position(3, 5, 6, time_point_number=5)
        other_pos = Position(0, 0, 0, time_point_number=5)

        positions_a = PositionCollection()
        positions_a.add(main_pos)
        positions_a.set_position_data(main_pos, "test_data_1", "foo")
        positions_a.set_position_data(main_pos, "test_data_2", "bar")

        # Of b, we overwrite test_data_1, and add test_data_3, and also add test_data_1 for a different position
        positions_b = PositionCollection()
        positions_b.add(main_pos)
        positions_b.add(other_pos)
        positions_b.set_position_data(main_pos, "test_data_1", "overwriting")
        positions_b.set_position_data(main_pos, "test_data_3", "baz")
        positions_b.set_position_data(other_pos, "test_data_1", "bam")

        # Now merge them, and check if the data is as expected
        positions_a.merge_data(positions_b)
        self.assertEqual("overwriting", positions_a.get_position_data(main_pos, "test_data_1"))
        self.assertEqual("bar", positions_a.get_position_data(main_pos, "test_data_2"))
        self.assertEqual("baz", positions_a.get_position_data(main_pos, "test_data_3"))
        self.assertEqual("bam", positions_a.get_position_data(other_pos, "test_data_1"))

        self.assertEqual({"test_data_1": str, "test_data_2": str, "test_data_3": str},
                         positions_a.get_data_names_and_types())
