from unittest import TestCase

import numpy

from organoid_tracker.core.images import Image
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import image_adder


class TestImageAdder(TestCase):

    def test_simple(self):
        # Creates a black square, draws a white squre inside it, leaving 1px black at the border
        zeros = Image(numpy.zeros((1, 10, 10), dtype=numpy.float32), Position(0, 0, 0))
        middle_stamp = Image(numpy.ones((1, 8, 8), dtype=numpy.float32), Position(1, 1, 0))

        image_adder.add_images(zeros, middle_stamp)

        self.assertEqual(0, zeros.value_at(Position(0, 0, 0)))
        self.assertEqual(1, zeros.value_at(Position(1, 1, 0)))
        self.assertEqual(1, zeros.value_at(Position(8, 8, 0)))
        self.assertEqual(0, zeros.value_at(Position(9, 9, 0)))

    def test_non_overlapping(self):
        zeros = Image(numpy.zeros((1, 10, 10), dtype=numpy.float32), Position(0, 0, 0))
        middle_stamp = Image(numpy.ones((1, 8, 8), dtype=numpy.float32), Position(11, 11, 0))

        image_adder.add_images(zeros, middle_stamp)

        self.assertEqual(0, zeros.value_at(Position(0, 0, 0)))
        self.assertEqual(0, zeros.value_at(Position(1, 1, 0)))
        self.assertEqual(0, zeros.value_at(Position(8, 8, 0)))
        self.assertEqual(0, zeros.value_at(Position(9, 9, 0)))

    def test_partially_overlapping(self):
        zeros = Image(numpy.zeros((1, 10, 10), dtype=numpy.float32), Position(0, 0, 0))
        middle_stamp = Image(numpy.ones((1, 8, 8), dtype=numpy.float32), Position(2, 2, 0))

        image_adder.add_images(zeros, middle_stamp)

        self.assertEqual(0, zeros.value_at(Position(0, 0, 0)))
        self.assertEqual(0, zeros.value_at(Position(1, 1, 0)))
        self.assertEqual(1, zeros.value_at(Position(8, 8, 0)))
        self.assertEqual(1, zeros.value_at(Position(9, 9, 0)))

    def test_summing(self):
        ones = Image(numpy.ones((1, 10, 10), dtype=numpy.float32), Position(0, 0, 0))
        middle_stamp = Image(numpy.ones((1, 8, 8), dtype=numpy.float32), Position(2, 2, 0))

        image_adder.add_images(ones, middle_stamp)  # Now most of the ones should become two

        self.assertEqual(1, ones.value_at(Position(0, 0, 0)))
        self.assertEqual(1, ones.value_at(Position(1, 1, 0)))
        self.assertEqual(2, ones.value_at(Position(8, 8, 0)))
        self.assertEqual(2, ones.value_at(Position(9, 9, 0)))
