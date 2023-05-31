import unittest
from organoid_tracker.imaging.image_file_name_pattern_finder import find_time_and_channel_pattern


class TestImageOffsets(unittest.TestCase):

    def test_time(self):
        self.assertEqual("image_{time}.png", find_time_and_channel_pattern(None, "image_1.png"))
        self.assertEqual("image_({time}).png", find_time_and_channel_pattern(None, "image_(1).png"))
        self.assertEqual("image_{time}.png", find_time_and_channel_pattern(None, "image_0.png"))
        self.assertEqual("image_{time:03}.png", find_time_and_channel_pattern(None, "image_001.png"))
        self.assertEqual("image_{time:02}.png", find_time_and_channel_pattern(None, "image_01.png"))
        self.assertEqual("nd799xy08t{time:03}.tif", find_time_and_channel_pattern(None, "nd799xy08t001.tif"))
        self.assertEqual(None, find_time_and_channel_pattern(None, "nd799xy08.tif"))  # Not a time lapse, so no pattern

    def test_time_and_channel(self):
        self.assertEqual("nd799xy08t{time:03}c{channel}.tif", find_time_and_channel_pattern(None, "nd799xy08t001c1.tif"))
        self.assertEqual("Mark_and_Find 001_Position001_C{channel}_T{time:02}.tif",
                          find_time_and_channel_pattern(None, "Mark_and_Find 001_Position001_C0_T00.tif"))
