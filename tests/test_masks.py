import unittest

import numpy

from autotrack.core.gaussian import Gaussian
from autotrack.core.mask import create_mask_for
from autotrack.core.shape import EllipseShape, GaussianShape, UnknownShape


class TestMasks(unittest.TestCase):

    def test_circle(self):
        numpy.random.seed(12)
        image = numpy.random.normal(loc=5,scale=2,size=(3,50,50))
        mask = create_mask_for(image)

        ellipse = UnknownShape()
        ellipse.draw_mask(mask, 23.01, 25, 1)
        masked_image = mask.create_masked_and_normalized_image(image)

        self.assertTrue(numpy.isnan(masked_image[0, 0, 0]))  # NaN values are used in corners

        # This point (at the center of the ellipse) must not be masked out
        self.assertFalse(numpy.isnan(masked_image[1 - mask.offset_z, 23 - mask.offset_y, 25 - mask.offset_x]))

    def test_ellipse(self):
        numpy.random.seed(12)
        image = numpy.random.normal(loc=5,scale=2,size=(5,30,30))
        mask = create_mask_for(image)

        ellipse = EllipseShape(0, 0, 21, 23, 30)
        ellipse.draw_mask(mask, 15.01, 15, 2)
        masked_image = mask.create_masked_and_normalized_image(image)

        self.assertEquals((1, 23, 21), masked_image.shape)  # Masked image must be cropped
        self.assertTrue(numpy.isnan(masked_image[0, 0, 0]))  # NaN values are used in corners

        # This point (at the center of the ellipse) must not be masked out
        self.assertFalse(numpy.isnan(masked_image[2 - mask.offset_z, 15 - mask.offset_y, 15 - mask.offset_x]))

    def test_gaussian(self):
        numpy.random.seed(13)
        image = numpy.random.normal(loc=5, scale=2, size=(16, 30, 30))
        Gaussian(30, 15, 15, 8, 3, 3, 3, 0, 0, 0).draw(image)
        mask = create_mask_for(image)

        gaussian = GaussianShape(Gaussian(30, 0, 0, 0, 3, 3, 3, 0, 0, 0))
        gaussian.draw_mask(mask, 15.01, 15, 8)
        masked_image = mask.create_masked_and_normalized_image(image)

        self.assertTrue(numpy.isnan(masked_image[0, 0, 0]))  # At edge there must be no value
        self.assertFalse(numpy.isnan(masked_image[8 - mask.offset_z, 15 - mask.offset_y, 15 - mask.offset_x]))
        # ^ At center there must be a value
