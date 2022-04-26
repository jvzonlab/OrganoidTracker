import unittest

from organoid_tracker.core import Name


class TestNameClass(unittest.TestCase):

    def test_basics(self):
        name = Name()

        name.set_name("Hi")
        self.assertEqual("Hi", str(name))

        name.set_name("Hello!")
        self.assertEqual("Hello!", str(name))

    def test_has_name(self):
        name = Name()
        self.assertFalse(name.has_name())

        name.set_name("Manually set")
        self.assertTrue(name.has_name())

    def test_path(self):
        name = Name()
        name.set_name("Fancy characters: /%$#=\\")
        self.assertEqual("Fancy characters_ _", name.get_save_name())

    def test_str(self):
        name = Name()

        name.set_name("Some test")
        self.assertEqual("Some test", str(name))

    def test_save_name(self):
        name = Name()
        name.set_name("My/Test")
        self.assertEqual("My_Test", name.get_save_name())
