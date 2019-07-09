import unittest

from autotrack.core.vector import Vector3
from autotrack.imaging import angles


class TestAngles(unittest.TestCase):

    def test_difference(self):
        # Some perpendicular angles - should be easy
        self.assertEquals(90, angles.difference(100, 10))
        self.assertEquals(90, angles.difference(280, 10))

        # These are a bit harder
        self.assertEquals(170, angles.difference(0, 170))
        self.assertEquals(175, angles.difference(355, 170))

    def test_change_direction(self):
        # Some basic tests
        self.assertEquals(100, angles.direction_change(0, 100))
        self.assertEquals(100, angles.direction_change(5, 105))
        self.assertEquals(100, angles.direction_change(355, 95))

        # Also one counter-clockwise
        self.assertEquals(-160, angles.direction_change(0, 200))

        # Some example values for an experiment
        self.assertEquals(-120, angles.direction_change(60, 300))
        self.assertEquals(-120, angles.direction_change(240, 120))
        self.assertEquals(60, angles.direction_change(240, 300))
        self.assertEquals(60, angles.direction_change(60, 120))

    def test_change_direction_with_flip(self):
        # Rotations to the right
        self.assertEquals(30, angles.direction_change_of_line(0, 30))
        self.assertEquals(0, angles.direction_change_of_line(0, 180))
        self.assertEquals(10, angles.direction_change_of_line(0, 190))
        self.assertEquals(20, angles.direction_change_of_line(170, 190))
        self.assertEquals(10, angles.direction_change_of_line(355, 5))

        # Rotations to the left
        self.assertEquals(30, angles.direction_change_of_line(30, 0))
        self.assertEquals(10, angles.direction_change_of_line(5, 355))

    def test_flipped(self):
        self.assertEquals(190, angles.flipped(10))
        self.assertEquals(10, angles.flipped(190))
        self.assertEquals(0, angles.flipped(180))

    def test_mirrored(self):
        self.assertEquals(100, angles.mirrored(80, mirror_angle=90))
        self.assertEquals(100, angles.mirrored(80, mirror_angle=270))

        self.assertEquals(10, angles.mirrored(350, mirror_angle=0))
        self.assertEquals(10, angles.mirrored(350, mirror_angle=180))

    def test_right_hand_rule(self):
        a = Vector3(1, 2, 3)
        b = Vector3(3, 2, 1)
        c = Vector3(1, 1, 1)
        self.assertAlmostEqual(129.23152048, angles.right_hand_rule(a, b, c))

    def test_right_hand_rule_on_straight_line(self):
        a = Vector3(1, 0, 0)
        b = Vector3(3, 0, 0)
        c = Vector3(5, 0, 0)
        self.assertAlmostEqual(180, angles.right_hand_rule(a, b, c))

    def test_right_hand_rule_difficult(self):
        # This one caused a crash earlier
        a = Vector3(71.68, 81.92, 10.0)
        b = Vector3(77.12, 88.96000000000001, 10.0)
        c = Vector3(81.63334669338678, 94.80080160320641, 10.0)
        self.assertAlmostEqual(180, angles.right_hand_rule(a, b, c))
