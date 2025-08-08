import unittest

from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position


class TestLinks(unittest.TestCase):

    def test_futures(self):
        position = Position(0, 0, 0, time_point_number=0)
        future_position = Position(1, 0, 0, time_point_number=1)
        links = Links()
        links.add_link(position, future_position)

        self.assertEqual({future_position}, links.find_futures(position))
        self.assertEqual(set(), links.find_futures(future_position))

    def test_pasts(self):
        position = Position(0, 0, 0, time_point_number=1)
        past_position = Position(1, 0, 0, time_point_number=0)
        links = Links()
        links.add_link(position, past_position)

        self.assertEqual({past_position}, links.find_pasts(position))
        self.assertEqual(set(), links.find_pasts(past_position))

    def test_move_in_time(self):
        links = Links()
        links.add_link(Position(0, 1, 2, time_point_number=0), Position(3, 4, 5, time_point_number=1))

        links.move_in_time(10)
        self.assertFalse(links.contains_link(
            Position(0, 1, 2, time_point_number=0), Position(3, 4, 5, time_point_number=1)))
        self.assertTrue(links.contains_link(
            Position(0, 1, 2, time_point_number=10), Position(3, 4, 5, time_point_number=11)))

    def test_has_link_data(self):
        links = Links()
        self.assertFalse(links.has_link_data())

        pos_1 = Position(0, 0, 0, time_point_number=0)
        pos_2 = Position(0, 0, 0, time_point_number=1)
        links.add_link(pos_1, pos_2)
        links.set_link_data(pos_1, pos_2, "test", "test_value")
        self.assertTrue(links.has_link_data())

        links.set_link_data(Position(0, 0, 0, time_point_number=0), Position(0, 0, 0, time_point_number=1), "test",
                                None)  # Removes this piece of data again
        self.assertFalse(links.has_link_data())

    def test_set_get_link_data(self):
        links = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        self.assertIsNone(links.get_link_data(pos1, pos2, "test"))

        links.add_link(pos1, pos2)
        links.set_link_data(pos1, pos2, "test", "test value")
        self.assertEqual("test value", links.get_link_data(pos1, pos2, "test"))

    def test_set_data_for_non_existing_link(self):
        """If the link does not exist, setting link data should silently fail."""
        links = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)

        links.set_link_data(pos1, pos2, "test", "test value")
        self.assertIsNone(links.get_link_data(pos1, pos2, "test"))

    def test_set_link_data_swap(self):
        links = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        links.add_link(pos1, pos2)
        self.assertIsNone(links.get_link_data(pos1, pos2, "test"))

        # Swap parameter order
        links.set_link_data(pos2, pos1, "test", "test value")

        # Try both orders
        self.assertEqual("test value", links.get_link_data(pos1, pos2, "test"))
        self.assertEqual("test value", links.get_link_data(pos2, pos1, "test"))

    def test_not_consecutive(self):
        links = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=0)  # Same time point, so not a valid link
        pos3 = Position(0, 0, 0, time_point_number=2)  # One time point skipped, so not a valid link

        with self.assertRaises(ValueError):
            links.get_link_data(pos1, pos2, "test")

        with self.assertRaises(ValueError):
            links.get_link_data(pos1, pos3, "test")

    def test_copy(self):
        links = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        links.add_link(pos1, pos2)
        links.set_link_data(pos1, pos2, "test", "old value")

        # Take a copy, then modify the original
        copy = links.copy()
        links.set_link_data(pos1, pos2, "test", "new value")

        # Make sure copy is not updated
        self.assertEqual("old value", copy.get_link_data(pos1, pos2, "test"))
        self.assertEqual("new value", links.get_link_data(pos1, pos2, "test"))

    def test_merge_data(self):
        links1 = Links()
        links2 = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        pos3 = Position(0, 0, 0, time_point_number=2)
        links1.add_link(pos1, pos2)
        links1.set_link_data(pos1, pos2, "test", "value1")
        links2.add_link(pos2, pos3)
        links2.set_link_data(pos2, pos3, "test", "value2")

        # Merge link_data1 and link_data2, check contents
        links1.merge_data(links2)
        self.assertEqual("value1", links1.get_link_data(pos1, pos2, "test"))
        self.assertEqual("value2", links1.get_link_data(pos2, pos3, "test"))

    def test_remove_link(self):
        links = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)

        links.add_link(pos1, pos2)
        links.set_link_data(pos1, pos2, "test", "test value")
        links.remove_link(pos1, pos2)
        self.assertIsNone(links.get_link_data(pos1, pos2, "test"))

    def test_replace_link(self):
        links = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        links.add_link(pos1, pos2)
        links.set_link_data(pos1, pos2, "test", "test value")

        pos3 = Position(10, 10, 10, time_point_number=0)
        pos4 = Position(10, 10, 10, time_point_number=1)

        links.replace_position(pos1, pos3)
        links.replace_position(pos2, pos4)
        self.assertIsNone(links.get_link_data(pos1, pos2, "test"))
        self.assertEqual("test value", links.get_link_data(pos4, pos3, "test"))

    def test_find_data_of_link(self):
        links = Links()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        pos_wrong = Position(0, 0, 0, time_point_number=2)

        # Set three data entries, but only two on the correct link
        links.add_link(pos1, pos2)
        links.add_link(pos2, pos_wrong)
        links.set_link_data(pos1, pos2, "test1", "test1 value")
        links.set_link_data(pos1, pos2, "test2", "test2 value")
        links.set_link_data(pos2, pos_wrong, "test2", "wrong value")

        # Check if we got the correct two data entries back
        found_link_data = dict(links.find_all_data_of_link(pos1, pos2))
        self.assertEqual("test1 value", found_link_data["test1"])
        self.assertEqual("test2 value", found_link_data["test2"])
        self.assertEqual(2, len(found_link_data))

    def test_move_data_in_time(self):
        links = Links()

        pos1 = Position(0, 1, 2, time_point_number=0)
        pos2 = Position(3, 4, 5, time_point_number=1)
        links.add_link(pos1, pos2)
        links.set_link_data(pos1, pos2, "test1", "test1 value")

        links.move_in_time(10)
        self.assertIsNone(links.get_link_data(
            Position(0, 1, 2, time_point_number=0), Position(3, 4, 5, time_point_number=1), "test1"))
        self.assertEqual("test1 value", links.get_link_data(
            Position(0, 1, 2, time_point_number=10), Position(3, 4, 5, time_point_number=11), "test1"))
