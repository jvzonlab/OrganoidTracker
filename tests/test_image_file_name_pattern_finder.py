import unittest
from organoid_tracker.imaging.image_file_name_pattern_finder import find_time_and_channel_pattern


class TestImageOffsets(unittest.TestCase):

    def test_time(self):
        self.assertEquals("image_{time}.png", find_time_and_channel_pattern("image_1.png"))
        self.assertEquals("image_({time}).png", find_time_and_channel_pattern("image_(1).png"))
        self.assertEquals("image_{time}.png", find_time_and_channel_pattern("image_0.png"))
        self.assertEquals("image_{time:03}.png", find_time_and_channel_pattern("image_001.png"))
        self.assertEquals("image_{time:02}.png", find_time_and_channel_pattern("image_01.png"))
        self.assertEquals("nd799xy08t{time:03}.tif", find_time_and_channel_pattern("nd799xy08t001.tif"))
        self.assertEquals(None, find_time_and_channel_pattern("nd799xy08.tif"))  # Not a time lapse, so no pattern

    def test_time_and_channel(self):
        self.assertEquals("nd799xy08t{time:03}c{channel}.tif", find_time_and_channel_pattern("nd799xy08t001c1.tif"))
        self.assertEquals("Mark_and_Find 001_Position001_C{channel}_T{time:02}.tif",
                          find_time_and_channel_pattern("Mark_and_Find 001_Position001_C0_T00.tif"))
