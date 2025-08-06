import unittest

from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.errors import Error


class TestLinkingMarkers(unittest.TestCase):

    def test_error_marker(self):
        positions = PositionCollection()

        self.assertEqual(None, linking_markers.get_error_marker(positions, Position(0, 0, 0, time_point_number=0)),
                          "non-existing position must have no error marker")

        position = Position(2, 2, 2, time_point_number=2)
        self.assertEqual(None, linking_markers.get_error_marker(positions, position), "no error marker was set")

        linking_markers.set_error_marker(positions, position, Error.MOVED_TOO_FAST)
        self.assertEqual(Error.MOVED_TOO_FAST, linking_markers.get_error_marker(positions, position))
        self.assertFalse(linking_markers.is_error_suppressed(positions, position, Error.MOVED_TOO_FAST))

        linking_markers.suppress_error_marker(positions, position, Error.MOVED_TOO_FAST)
        self.assertEqual(None, linking_markers.get_error_marker(positions, position), "error must be suppressed")
        self.assertTrue(linking_markers.is_error_suppressed(positions, position, Error.MOVED_TOO_FAST))
