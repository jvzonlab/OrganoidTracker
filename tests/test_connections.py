import unittest

from organoid_tracker.core import TimePoint
from organoid_tracker.core.connections import Connections
from organoid_tracker.core.position import Position


class TestConnections(unittest.TestCase):

    def test_add_and_exists(self):
        pos1 = Position(2, 3, 4, time_point_number=5)
        pos2 = Position(1, 3, 4, time_point_number=5)
        pos3 = Position(5, 3, 4, time_point_number=5)

        connections = Connections()
        connections.add_connection(pos1, pos2)

        self.assertTrue(connections.contains_connection(pos1, pos2))
        self.assertTrue(connections.contains_connection(pos2, pos1))
        self.assertFalse(connections.contains_connection(pos1, pos3))
        self.assertFalse(connections.contains_connection(pos3, pos1))

    def test_add_different_time_points(self):
        pos1 = Position(2, 3, 4, time_point_number=5)
        pos2 = Position(1, 3, 4, time_point_number=6)

        connections = Connections()

        # Should fail, as pos1 and pos2 are in different time points
        self.assertRaises(ValueError, lambda: connections.add_connection(pos1, pos2))
        self.assertFalse(connections.contains_connection(pos1, pos2))  # And no connection must have been made

    def test_add_no_time_point(self):
        pos1 = Position(2, 3, 4)
        pos2 = Position(1, 3, 4)

        connections = Connections()

        # Should fail, as no time point was specified
        self.assertRaises(ValueError, lambda: connections.add_connection(pos1, pos2))
        self.assertFalse(connections.contains_connection(pos1, pos2))  # And no connection must have been made

    def test_copy(self):
        pos1 = Position(2, 3, 4, time_point_number=3)
        pos2 = Position(1, 3, 4, time_point_number=3)

        connections_1 = Connections()
        connections_1.add_connection(pos1, pos2)

        # Test the copy
        connections_2 = connections_1.copy()
        self.assertTrue(connections_1.contains_connection(pos1, pos2))
        self.assertTrue(connections_2.contains_connection(pos1, pos2))

        # Modify the original, test whether copy is unaffected
        connections_1.remove_connection(pos1, pos2)
        self.assertFalse(connections_1.contains_connection(pos1, pos2))
        self.assertTrue(connections_2.contains_connection(pos1, pos2))

    def test_move_in_time(self):
        connections = Connections()
        connections.add_connection(Position(2, 3, 4, time_point_number=3), Position(1, 3, 4, time_point_number=3))

        connections.move_in_time(10)

        self.assertFalse(connections.contains_connection(
            Position(2, 3, 4, time_point_number=3), Position(1, 3, 4, time_point_number=3)))
        self.assertTrue(connections.contains_connection(
            Position(2, 3, 4, time_point_number=13), Position(1, 3, 4, time_point_number=13)))

    def test_connection_data(self):
        connections = Connections()
        pos_a = Position(2, 3, 4, time_point_number=3)
        pos_b = Position(1, 3, 4, time_point_number=3)

        # Check if there's no data yet (should always return None for non-existing connections)
        self.assertIsNone(connections.get_data_of_connection(pos_a, pos_b, "test_key"))
        # Now add the connection (without any data)
        connections.add_connection(pos_a, pos_b)
        # Check if it still returns None (we didn't add any data yet)
        self.assertIsNone(connections.get_data_of_connection(pos_a, pos_b, "test_key"))
        # Now set some data, and check
        connections.set_data_of_connection(pos_a, pos_b, "test_key", "test_value")
        self.assertEqual("test_value", connections.get_data_of_connection(pos_a, pos_b, "test_key"))
        # Test if it still works if we swap the arguments pos_a and pos_b
        self.assertEqual("test_value", connections.get_data_of_connection(pos_b, pos_a, "test_key"))

        # Check metadata keys
        self.assertEqual({"test_key"}, connections.find_all_data_names())

        # Remove the data, and check again
        connections.set_data_of_connection(pos_b, pos_a, "test_key", None)
        self.assertEqual(set(), connections.find_all_data_names())
