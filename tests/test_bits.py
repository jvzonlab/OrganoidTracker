from unittest import TestCase

import numpy

from organoid_tracker.util import bits


class Test(TestCase):

    def test_ensure_8bit(self):
        # Make sure 8bit image is not converted
        already_8bit = numpy.zeros((4,4,4), dtype=numpy.uint8)
        self.assertIs(already_8bit, bits.ensure_8bit(already_8bit))

        # Make sure 16-bit image is converted
        image_16bit = numpy.zeros((4,4,4), dtype=numpy.uint16)
        converted = bits.ensure_8bit(image_16bit)
        self.assertEqual(numpy.uint8, converted.dtype)

        # Make sure images aren't converted twice
        self.assertIs(converted, bits.ensure_8bit(converted))
