import unittest

from autotrack.particle_detection.ellipse_cluster import Ellipse

class TestFindNearestFew(unittest.TestCase):

    def test_basic(self):
        a = Ellipse(320.19622802734375, 256.3709411621094, 8.912117195129394, 11.489200210571289, 137.1503143310547)
        b = Ellipse(311.13555908203125, 275.9170837402344, 11.996450805664061, 14.559389305114745, 22.218984603881836)

        self.assertFalse(a.intersects(b))
