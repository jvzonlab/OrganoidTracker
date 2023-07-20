import unittest

from organoid_tracker.gui.location_map import LocationMap


class TestLocationMap(unittest.TestCase):

    def test_add_single(self):
        location_map = LocationMap()
        location_map.set(10, 20, "test")
        self.assertEqual("test", location_map.get_nearby(10, 20))
        self.assertEqual(None, location_map.get_nearby(0, 0))
        self.assertEqual(None, location_map.get_nearby(100, 100))

    def test_add_area(self):
        location_map = LocationMap()
        location_map.set_area(5, 2, 60, 80, "test")
        self.assertEqual("test", location_map.get_nearby(30, 30))
        self.assertEqual(None, location_map.get_nearby(100, 100))

    def test_resizing(self):
        location_map = LocationMap()
        location_map.set(50, 51, "test")
        location_map.set(200, 300, "test2")  # Needs to resize to fit this
        self.assertEqual("test", location_map.get_nearby(50, 51))  # Make sure original locations are unaffected

    def test_search(self):
        location_map = LocationMap()
        location_map.set(50, 51, "test")
        location_map.set(200, 300, "test2")

        # Find one object
        x, y = location_map.find_object("test2")
        self.assertEqual(200, x)
        self.assertEqual(300, y)

        # Test non-existing object
        x, y = location_map.find_object("test-non-existing")
        self.assertEqual(None, x)
        self.assertEqual(None, y)
