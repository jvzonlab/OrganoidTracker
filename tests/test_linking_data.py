from unittest import TestCase

from ai_track.core.links import Links
from ai_track.core.position import Position


class TestLinkingData(TestCase):

    def test_has_position_data(self):
        links = Links()
        position = Position(3, 5, 6, time_point_number=5)
        self.assertFalse(links.has_position_data_with_name("test_data"))
        links.set_position_data(position, "test_data", True)
        self.assertTrue(links.has_position_data_with_name("test_data"))
        links.set_position_data(position, "test_data", None)
        self.assertFalse(links.has_position_data_with_name("test_data"))
