from unittest import TestCase

from organoid_tracker.core.link_data import LinkData
from organoid_tracker.core.position import Position


class TestLinkData(TestCase):

    def test_has_link_data(self):
        link_data = LinkData()
        self.assertFalse(link_data.has_link_data())

        link_data.set_link_data(Position(0, 0, 0, time_point_number=0), Position(0, 0, 0, time_point_number=1), "test",
                                "test_value")
        self.assertTrue(link_data.has_link_data())

        link_data.set_link_data(Position(0, 0, 0, time_point_number=0), Position(0, 0, 0, time_point_number=1), "test",
                                None)  # Removes this piece of data again
        self.assertFalse(link_data.has_link_data())

    def test_set_get_link_data(self):
        link_data = LinkData()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        self.assertIsNone(link_data.get_link_data(pos1, pos2, "test"))

        link_data.set_link_data(pos1, pos2, "test", "test value")
        self.assertEqual("test value", link_data.get_link_data(pos1, pos2, "test"))

    def test_set_link_data_swap(self):
        link_data = LinkData()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        self.assertIsNone(link_data.get_link_data(pos1, pos2, "test"))

        # Swap parameter order
        link_data.set_link_data(pos2, pos1, "test", "test value")

        # Try both orders
        self.assertEqual("test value", link_data.get_link_data(pos1, pos2, "test"))
        self.assertEqual("test value", link_data.get_link_data(pos2, pos1, "test"))

    def test_not_consecutive(self):
        link_data = LinkData()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=0)  # Same time point, so not a valid link
        pos3 = Position(0, 0, 0, time_point_number=2)  # One time point skipped, so not a valid link

        with self.assertRaises(ValueError):
            link_data.get_link_data(pos1, pos2, "test")

        with self.assertRaises(ValueError):
            link_data.get_link_data(pos1, pos3, "test")

    def test_copy(self):
        link_data = LinkData()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        link_data.set_link_data(pos1, pos2, "test", "old value")

        # Take a copy, then modify the original
        copy = link_data.copy()
        link_data.set_link_data(pos1, pos2, "test", "new value")

        # Make sure copy is not updated
        self.assertEqual("old value", copy.get_link_data(pos1, pos2, "test"))
        self.assertEqual("new value", link_data.get_link_data(pos1, pos2, "test"))

    def test_merge_data(self):
        link_data1 = LinkData()
        link_data2 = LinkData()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        pos3 = Position(0, 0, 0, time_point_number=2)
        link_data1.set_link_data(pos1, pos2, "test", "value1")
        link_data2.set_link_data(pos2, pos3, "test", "value2")

        # Merge link_data1 and link_data2, check contents
        link_data1.merge_data(link_data2)
        self.assertEqual("value1", link_data1.get_link_data(pos1, pos2, "test"))
        self.assertEqual("value2", link_data1.get_link_data(pos2, pos3, "test"))

    def test_remove_link(self):
        link_data = LinkData()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)

        link_data.set_link_data(pos1, pos2, "test", "test value")
        link_data.remove_link(pos1, pos2)
        self.assertIsNone(link_data.get_link_data(pos1, pos2, "test"))

    def test_replace_link(self):
        link_data = LinkData()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        link_data.set_link_data(pos1, pos2, "test", "test value")

        pos3 = Position(10, 10, 10, time_point_number=1)
        pos4 = Position(10, 10, 10, time_point_number=0)

        link_data.replace_link(pos1, pos2, pos3, pos4)
        self.assertIsNone(link_data.get_link_data(pos1, pos2, "test"))
        self.assertEqual("test value", link_data.get_link_data(pos3, pos4, "test"))

    def test_find_data_of_link(self):
        link_data = LinkData()
        pos1 = Position(0, 0, 0, time_point_number=0)
        pos2 = Position(0, 0, 0, time_point_number=1)
        pos_wrong = Position(0, 0, 0, time_point_number=2)

        # Set three data entries, but only two on the correct link
        link_data.set_link_data(pos1, pos2, "test1", "test1 value")
        link_data.set_link_data(pos1, pos2, "test2", "test2 value")
        link_data.set_link_data(pos2, pos_wrong, "test2", "wrong value")

        # Check if we got the correct two data entries back
        found_link_data = dict(link_data.find_all_data_of_link(pos1, pos2))
        self.assertEqual("test1 value", found_link_data["test1"])
        self.assertEqual("test2 value", found_link_data["test2"])
        self.assertEqual(2, len(found_link_data))

    def test_move_in_time(self):
        link_data = LinkData()
        link_data.set_link_data(Position(0, 1, 2, time_point_number=0), Position(3, 4, 5, time_point_number=1),
                                "test1", "test1 value")

        link_data.move_in_time(10)
        self.assertIsNone(link_data.get_link_data(
            Position(0, 1, 2, time_point_number=0), Position(3, 4, 5, time_point_number=1), "test1"))
        self.assertEqual("test1 value", link_data.get_link_data(
            Position(0, 1, 2, time_point_number=10), Position(3, 4, 5, time_point_number=11), "test1"))
