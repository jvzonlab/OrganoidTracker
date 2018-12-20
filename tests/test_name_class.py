import unittest

from autotrack.core import Name


class TestNameClass(unittest.TestCase):

    def test_basics(self):
        name = Name()

        name.set_name("Hi")
        self.assertEquals("Hi", str(name))

        name.set_name("Hello!")
        self.assertEquals("Hello!", str(name))

    def test_has_name(self):
        name = Name()
        self.assertFalse(name.has_name())

        name.set_name("Manually set")
        self.assertTrue(name.has_name())

    def test_path(self):
        name = Name()
        name.set_name("Fancy characters: /%$#=\\")
        self.assertEquals("Fancy characters_ _", name.get_save_name())

    def test_str(self):
        name = Name()

        name.set_name("Some test")
        self.assertEquals("Some test", str(name))

    def test_save_name(self):
        name = Name()
        name.set_name("My/Test")
        self.assertEquals("My_Test", name.get_save_name())
