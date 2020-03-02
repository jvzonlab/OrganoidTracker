import unittest

from organoid_tracker.core import min_none, max_none


class TestMinMax(unittest.TestCase):

    def test_min_list(self):
        self.assertEquals(2, min_none([3, 4, None, 2, 8]))
        self.assertEquals(None, min_none([None, None]))
        self.assertEquals(None, min_none([]))

    def test_min_args(self):
        self.assertEquals(2, min_none(3, 4, None, 2, 8))
        self.assertEquals(None, min_none(None, None))
        self.assertEquals(None, min_none(None))

    def test_max_list(self):
        self.assertEquals(8, max_none([3, 4, None, 2, 8]))
        self.assertEquals(None, max_none([None, None]))
        self.assertEquals(None, max_none([]))

    def test_max_args(self):
        self.assertEquals(8, max_none(3, 4, None, 2, 8))
        self.assertEquals(None, max_none(None, None))
        self.assertEquals(None, max_none(None))
