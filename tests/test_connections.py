import unittest

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
