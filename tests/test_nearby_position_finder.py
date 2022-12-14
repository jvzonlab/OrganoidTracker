import unittest

from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.linking import nearby_position_finder

class TestNearbyPositionFinder(unittest.TestCase):

    def test_make_nearby_positions_graph(self):
        positions = [
            Position(10, 0, 0),
            Position(11, 0, 0),
            Position(12, 0, 0),
            Position(13, 0, 0),
            Position(14, 0, 0),
            Position(14, 1, 0),
            Position(14, 2, 0),
            Position(14, 3, 0)
        ]

        # Find the two nearest neighbors
        graph = nearby_position_finder.make_nearby_positions_graph(ImageResolution.PIXELS, positions, neighbors=2)

        # Test connections
        self.assertTrue(graph.has_edge(Position(11, 0, 0), Position(10, 0, 0)))
        self.assertTrue(graph.has_edge(Position(11, 0, 0), Position(12, 0, 0)))
        self.assertFalse(graph.has_edge(Position(11, 0, 0), Position(13, 0, 0)))

        # Test distances
        self.assertEqual(1, graph[Position(11, 0, 0)][Position(12, 0, 0)]["distance_um"])
