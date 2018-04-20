import unittest

from imaging import stacked_image
import numpy


class TestImageAverage(unittest.TestCase):

    def test_simple_case(self):
        input = numpy.array([
            [0, 0, 0, 0],
            [3, 3, 3, 3],
            [0, 0, 0, 0],
            [3, 3, 3, 3],
            [0, 0, 0, 0],
        ], dtype=numpy.float32)
        expected = numpy.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ], dtype=numpy.float32)
        output = stacked_image.average_out(input, num_layers=1)
        self.assertTrue(numpy.array_equal(expected, output))

    def test_2d_arrays(self):
        input = numpy.array([
            [[0, 0], [0, 0]],
            [[3, 3], [3, 3]],
            [[0, 0], [0, 0]],
            [[3, 3], [3, 3]],
            [[0, 0], [0, 0]],
        ], dtype=numpy.float32)
        expected = numpy.array([
            [[0, 0], [0, 0]],
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
            [[1, 1], [1, 1]],
            [[0, 0], [0, 0]],
        ], dtype=numpy.float32)
        output = stacked_image.average_out(input, num_layers=1)
        self.assertTrue(numpy.array_equal(expected, output))
