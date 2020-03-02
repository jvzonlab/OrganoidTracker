import unittest

from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position


class TestLinks(unittest.TestCase):

    def test_data(self):
        position = Position(0, 0, 0, time_point_number=0)
        links = Links()
        links.set_position_data(position, "name", "AA")

        self.assertEquals("AA", links.get_position_data(position, "name"))

    def test_futures(self):
        position = Position(0, 0, 0, time_point_number=0)
        future_position = Position(1, 0, 0, time_point_number=1)
        links = Links()
        links.add_link(position, future_position)

        self.assertEquals({future_position}, links.find_futures(position))
        self.assertEquals(set(), links.find_futures(future_position))

    def test_pasts(self):
        position = Position(0, 0, 0, time_point_number=1)
        past_position = Position(1, 0, 0, time_point_number=0)
        links = Links()
        links.add_link(position, past_position)

        self.assertEquals({past_position}, links.find_pasts(position))
        self.assertEquals(set(), links.find_pasts(past_position))
