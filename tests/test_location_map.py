import unittest

from autotrack.gui.location_map import LocationMap


class TestLocationMap(unittest.TestCase):

    def test_add_single(self):
        location_map = LocationMap()
        location_map.set(10, 20, "test")
        self.assertEquals("test", location_map.get_nearby(10, 20))
        self.assertEquals(None, location_map.get_nearby(0, 0))
        self.assertEquals(None, location_map.get_nearby(100, 100))

    def test_add_area(self):
        location_map = LocationMap()
        location_map.set_area(5, 2, 60, 80, "test")
        self.assertEquals("test", location_map.get_nearby(30, 30))
        self.assertEquals(None, location_map.get_nearby(100, 100))

    def test_resizing(self):
        location_map = LocationMap()
        location_map.set(50, 51, "test")
        location_map.set(200, 300, "test2")  # Needs to resize to fit this
        self.assertEquals("test", location_map.get_nearby(50, 51))  # Make sure original locations are unaffected
