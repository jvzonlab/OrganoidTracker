import numpy
import unittest

from organoid_tracker.core.gaussian import Gaussian


class TestGaussians(unittest.TestCase):

    def test_gradient(self):
        gaussian = Gaussian(a=3, mu_x=10, mu_y=10, mu_z=10, cov_xx=2, cov_yy=2, cov_zz=2, cov_xy=0, cov_xz=0, cov_yz=0)

        # At center most gradients must be zero
        images = []
        for i in range(10):
            image = numpy.zeros((30, 512, 512), dtype=numpy.float64)
            gaussian.draw_gradient(image, i)
            images.append(image)

        # At center most gradients must be zero
        self.assertTrue(images[0][10, 10, 10] == 1)
        self.assertTrue(images[1][10, 10, 10] == 0)
        self.assertTrue(images[2][10, 10, 10] == 0)
        self.assertTrue(images[3][10, 10, 10] == 0)

        # A bit from the center that changes
        self.assertTrue(images[0][10, 10, 9] > 0)  # dG/da = G/a (just like d(3x)/dx = 3x/x = 3)
        self.assertTrue(images[1][10, 10, 9] < 0)  # dG/d(mu_x)
        self.assertTrue(images[2][10, 10, 9] == 0)  # dG/d(mu_y)
        self.assertTrue(images[3][10, 10, 9] == 0)  # dG/d(mu_z)
        self.assertTrue(images[4][10, 10, 9] > 0)  # dG/d(cov_xx)
        self.assertTrue(images[5][10, 10, 9] == 0)  # dG/d(cov_yy)
        self.assertTrue(images[6][10, 10, 9] == 0)  # dG/d(cov_zz)
        self.assertTrue(images[7][10, 10, 9] == 0)  # dG/d(cov_xy)
        self.assertTrue(images[8][10, 10, 9] == 0)  # dG/d(cov_xz)
        self.assertTrue(images[9][10, 10, 9] == 0)  # dG/d(cov_yz)
