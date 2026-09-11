import unittest

import numpy
from organoid_tracker.imaging.cropper import crop_3d


class TestDataAxis(unittest.TestCase):

    def test_negative_z(self):

        # image has 124 z-planes, 64 rows (y), 47 cols (x)
        image = numpy.zeros((124, 64, 47), dtype=numpy.uint8)

        # output has 62 z-planes, 64 rows, 47 cols
        output = numpy.zeros((62, 64, 47), dtype=numpy.uint8)

        # Choose a large negative z_start so the code path that adjusts output_z_offset runs
        x_start = 0
        y_start = 0
        z_start = -124

        # This call reproduces the ValueError:
        crop_3d(image, x_start, y_start, z_start, output)