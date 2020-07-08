import unittest

from organoid_tracker.config import config_type_str


class TestConfig(unittest.TestCase):

    def test_strip_quotes(self):
        """People often leave the quotes in the settings file. Just accept it and remove them."""
        self.assertEqual("test", config_type_str("test"))
        self.assertEqual("test", config_type_str("'test'"))
        self.assertEqual("test", config_type_str("\"test\""))
        self.assertEqual("\'test\"", config_type_str("\'test\""))  # Invalid quotes
        self.assertEqual("'", config_type_str("'"))  # Invalid quotes

