import unittest

from autotrack.core import Name


class TestNameClass(unittest.TestCase):

    def test_basics(self):
        name = Name()

        name.set_name("Hi")
        self.assertEquals("Hi", str(name))

        name.set_name("Hello!")
        self.assertEquals("Hello!", str(name))

    def test_overwrite(self):
        name = Name()

        name.provide_automatic_name("Test")
        self.assertEquals("Test", str(name))

        name.set_name("Hi!")
        self.assertEquals("Hi!", str(name))

    def test_no_overwrite(self):
        name = Name()

        name.set_name("Manually set")
        self.assertEquals("Manually set", str(name))

        name.provide_automatic_name("Automatically determined")
        self.assertEquals("Manually set", str(name))  # Don't set automatic names override manual names

    def test_path(self):
        name = Name()
        name.set_name("Fancy characters: /%$#=\\")
        self.assertEquals("Fancy characters_ _", name.get_save_name())