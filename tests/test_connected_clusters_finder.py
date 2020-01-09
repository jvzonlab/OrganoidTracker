import unittest

from ai_track.connecting import cluster_finder
from ai_track.core import TimePoint
from ai_track.core.connections import Connections
from ai_track.core.position import Position
from ai_track.core.position_collection import PositionCollection


class TestAngles(unittest.TestCase):

    def test_find_clusters(self):
        """Test whether two clusters of three positions each are correctly formed."""
        connections = Connections()

        # Set up cluster ABC
        position_a = Position(0, 0, 0, time_point_number=1)
        position_b = Position(0, 0, 1, time_point_number=1)
        position_c = Position(0, 0, 2, time_point_number=1)
        connections.add_connection(position_a, position_b)
        connections.add_connection(position_b, position_c)

        # Set up cluster XYZ
        position_x = Position(1, 0, 0, time_point_number=1)
        position_y = Position(1, 0, 1, time_point_number=1)
        position_z = Position(1, 0, 2, time_point_number=1)
        connections.add_connection(position_x, position_y)
        connections.add_connection(position_y, position_z)

        # Register those positions
        positions = PositionCollection((position_a, position_b, position_c, position_x, position_y, position_z))

        # Test if there are two clusters found
        clusters = cluster_finder.find_clusters(positions, connections, TimePoint(1))
        self.assertEquals(2, len(clusters))

        # Find the ABC and XYZ cluster
        cluster_abc, cluster_xyz = clusters  # Assumes that cluster ABC is the first cluster
        if position_a in cluster_xyz.positions:  # Corrects for when the above assumption is not ture
            cluster_abc, cluster_xyz = cluster_xyz, cluster_abc

        # Check cluster contents
        self.assertEquals({position_a, position_b, position_c}, cluster_abc.positions)
        self.assertEquals({position_x, position_y, position_z}, cluster_xyz.positions)

    def test_lone_cluster(self):
        """Test whether positions without a connection get put in their own cluster."""
        position = Position(0, 0, 0, time_point_number=1)
        connections = Connections()
        positions = PositionCollection((position,))

        clusters = cluster_finder.find_clusters(positions, connections, TimePoint(1))
        self.assertEquals(1, len(clusters))
        self.assertEquals({position}, clusters[0].positions)
