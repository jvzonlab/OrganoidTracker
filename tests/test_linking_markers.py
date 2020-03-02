import unittest

from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.errors import Error


class TestLinkingMarkers(unittest.TestCase):

    def test_error_marker(self):
        position_links = Links()

        self.assertEquals(None, linking_markers.get_error_marker(position_links, Position(0, 0, 0, time_point_number=0)),
                          "non-existing position must have no error marker")

        position = Position(2, 2, 2, time_point_number=2)
        self.assertEquals(None, linking_markers.get_error_marker(position_links, position), "no error marker was set")

        linking_markers.set_error_marker(position_links, position, Error.MOVED_TOO_FAST)
        self.assertEquals(Error.MOVED_TOO_FAST, linking_markers.get_error_marker(position_links, position))
        self.assertFalse(linking_markers.is_error_suppressed(position_links, position, Error.MOVED_TOO_FAST))

        linking_markers.suppress_error_marker(position_links, position, Error.MOVED_TOO_FAST)
        self.assertEquals(None, linking_markers.get_error_marker(position_links, position), "error must be suppressed")
        self.assertTrue(linking_markers.is_error_suppressed(position_links, position, Error.MOVED_TOO_FAST))
