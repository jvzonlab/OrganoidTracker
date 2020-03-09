import math
import unittest

from organoid_tracker.coordinate_system.spherical_coordinates import SphericalCoordinate
from organoid_tracker.core.vector import Vector3


class TestSphericalCoordinates(unittest.TestCase):

    def test_from_cartesian(self):
        vector = Vector3(0, -4, -4)
        spherical = SphericalCoordinate.from_cartesian(vector)
        self.assertEqual(SphericalCoordinate(math.sqrt(32), 135, 270), spherical)

    def test_to_cartesian(self):
        spherical = SphericalCoordinate(math.sqrt(32), 135, 270)
        self.assertEqual(Vector3(0, -4, -4), spherical.to_cartesian())

    def test_roundtrip(self):
        vector = Vector3(11.801400599195677, 2.174153604808981, 24.012673613516096)
        self.assertEqual(vector, SphericalCoordinate.from_cartesian(vector).to_cartesian())

    def test_zero_cartesian(self):
        """Makes sure there are no divisions by zero."""
        vector = Vector3(0, 0, 0)
        spherical = SphericalCoordinate.from_cartesian(vector)
        self.assertEqual(SphericalCoordinate(0, 0, 0), spherical)
