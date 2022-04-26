import unittest

from organoid_tracker.core.vector import Vector3
from organoid_tracker.imaging import angles


class TestAngles(unittest.TestCase):

    def test_difference(self):
        # Some perpendicular angles - should be easy
        self.assertEqual(90, angles.difference(100, 10))
        self.assertEqual(90, angles.difference(280, 10))

        # These are a bit harder
        self.assertEqual(170, angles.difference(0, 170))
        self.assertEqual(175, angles.difference(355, 170))

    def test_change_direction(self):
        # Some basic tests
        self.assertEqual(100, angles.direction_change(0, 100))
        self.assertEqual(100, angles.direction_change(5, 105))
        self.assertEqual(100, angles.direction_change(355, 95))

        # Also one counter-clockwise
        self.assertEqual(-160, angles.direction_change(0, 200))

        # Some example values for an experiment
        self.assertEqual(-120, angles.direction_change(60, 300))
        self.assertEqual(-120, angles.direction_change(240, 120))
        self.assertEqual(60, angles.direction_change(240, 300))
        self.assertEqual(60, angles.direction_change(60, 120))

    def test_change_direction_with_flip(self):
        # Rotations to the right
        self.assertEqual(30, angles.direction_change_of_line(0, 30))
        self.assertEqual(0, angles.direction_change_of_line(0, 180))
        self.assertEqual(10, angles.direction_change_of_line(0, 190))
        self.assertEqual(20, angles.direction_change_of_line(170, 190))
        self.assertEqual(10, angles.direction_change_of_line(355, 5))

        # Rotations to the left
        self.assertEqual(30, angles.direction_change_of_line(30, 0))
        self.assertEqual(10, angles.direction_change_of_line(5, 355))

    def test_flipped(self):
        self.assertEqual(190, angles.flipped(10))
        self.assertEqual(10, angles.flipped(190))
        self.assertEqual(0, angles.flipped(180))

    def test_mirrored(self):
        self.assertEqual(100, angles.mirrored(80, mirror_angle=90))
        self.assertEqual(100, angles.mirrored(80, mirror_angle=270))

        self.assertEqual(10, angles.mirrored(350, mirror_angle=0))
        self.assertEqual(10, angles.mirrored(350, mirror_angle=180))

    def test_right_hand_rule(self):
        a = Vector3(1, 2, 3)
        b = Vector3(3, 2, 1)
        c = Vector3(1, 1, 1)
        self.assertAlmostEqual(129.23152048, angles.right_hand_rule(a, b, c))

    def test_right_hand_rule_on_straight_line(self):
        a = Vector3(1, 0, 0)
        b = Vector3(3, 0, 0)
        c = Vector3(5, 0, 0)
        self.assertAlmostEqual(0, angles.right_hand_rule(a, b, c))

    def test_right_hand_rule_difficult(self):
        # This one caused a crash earlier
        a = Vector3(71.68, 81.92, 10.0)
        b = Vector3(77.12, 88.96000000000001, 10.0)
        c = Vector3(81.63334669338678, 94.80080160320641, 10.0)
        self.assertAlmostEqual(0, angles.right_hand_rule(a, b, c))
