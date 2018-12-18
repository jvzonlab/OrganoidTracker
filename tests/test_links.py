import unittest

from autotrack.core.links import PositionLinks
from autotrack.core.positions import Position


class TestLinks(unittest.TestCase):

    def test_data(self):
        position = Position(0, 0, 0).with_time_point_number(0)
        links = PositionLinks()
        links.set_position_data(position, "name", "AA")

        self.assertEquals("AA", links.get_position_data(position, "name"))

    def test_futures(self):
        position = Position(0, 0, 0).with_time_point_number(0)
        future_position = Position(1, 0, 0).with_time_point_number(1)
        links = PositionLinks()
        links.add_link(position, future_position)

        self.assertEquals({future_position}, links.find_futures(position))
        self.assertEquals(set(), links.find_futures(future_position))

    def test_pasts(self):
        position = Position(0, 0, 0).with_time_point_number(1)
        past_position = Position(1, 0, 0).with_time_point_number(0)
        links = PositionLinks()
        links.add_link(position, past_position)

        self.assertEquals({past_position}, links.find_pasts(position))
        self.assertEquals(set(), links.find_pasts(past_position))
