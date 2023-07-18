import unittest

from organoid_tracker.core.global_data import GlobalData


class TestGlobalData(unittest.TestCase):

    def test_simple(self):
        global_data = GlobalData()
        global_data.set_data("name", "AA")

        self.assertEquals("AA", global_data.get_data("name"))

    def test_initialization(self):
        global_data = GlobalData({"name": "BB"})

        self.assertEquals("BB", global_data.get_data("name"))

    def test_get_all(self):
        global_data = GlobalData()
        global_data.set_data("name", "CC")

        self.assertEqual({"name": "CC"}, global_data.get_all_data())

    def test_copy(self):
        global_data = GlobalData()
        global_data.set_data("name", "DD")

        copy = global_data.copy()
        copy.set_data("name", "EE")

        # Make sure changes to the copy don't write through
        self.assertEqual("DD", global_data.get_data("name"))
        self.assertEqual("EE", copy.get_data("name"))
