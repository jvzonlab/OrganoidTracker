import scipy.optimize
import numpy, numpy.linalg
from numpy import ndarray
import mahotas.labeled
from typing import List


class Gaussian:
    max: float
    mu_x: float
    mu_y: float
    mu_z: float
    cov_xx: float
    cov_yy: float
    cov_zz: float
    cov_xy: float
    cov_xz: float
    cov_yz: float

    _draw_xrange: int
    _draw_yrange: int
    _draw_zrange: int

    def __init__(self, max, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
        """Creates a Gaussian. All parameters are floats."""
        self.max = max
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.mu_z = mu_z
        self.cov_xx = cov_xx
        self.cov_yy = cov_yy
        self.cov_zz = cov_zz
        self.cov_xy = cov_xy
        self.cov_xz = cov_xz
        self.cov_yz = cov_yz

        # Draw to 3 times the standard deviation; this makes sure that 99.7% of the distribution is drawn
        self._draw_xrange = numpy.ceil(3 * numpy.sqrt(self.cov_xx))
        self._draw_yrange = numpy.ceil(3 * numpy.sqrt(self.cov_yy))
        self._draw_zrange = numpy.ceil(3 * numpy.sqrt(self.cov_zz))

    def add_to_image(self, image: ndarray):
        """Draws the Gaussian in the image."""
        start_x = max(0, int(self.mu_x - self._draw_xrange))
        start_y = max(0, int(self.mu_y - self._draw_yrange))
        start_z = max(0, int(self.mu_z - self._draw_zrange))
        image_view = image[
                     start_z:int(self.mu_z + self._draw_zrange + 1),
                     start_y:int(self.mu_y + self._draw_yrange + 1),
                     start_x:int(self.mu_x + self._draw_xrange + 1)
        ]

        mu = numpy.array([self.mu_x, self.mu_y, self.mu_z])
        covariance_matrix = numpy.array([
            [self.cov_xx, self.cov_xy, self.cov_xz],
            [self.cov_xy, self.cov_yy, self.cov_yz],
            [self.cov_xz, self.cov_yz, self.cov_zz]
        ], dtype=numpy.float32)
        covariance_matrix_inversed = numpy.linalg.inv(covariance_matrix)

        pos_xyz = numpy.empty(3, dtype=numpy.uint32)
        for pos_zyx_offset in numpy.ndindex(image_view.shape):
            pos_xyz[0] = pos_zyx_offset[2] + start_x
            pos_xyz[1] = pos_zyx_offset[1] + start_y
            pos_xyz[2] = pos_zyx_offset[0] + start_z
            image_view[pos_zyx_offset] += _gaussian_3d_single(pos_xyz, self.max, mu, covariance_matrix_inversed)


def fit_gaussians(blurred_image: ndarray, watershedded_image: ndarray, buffer_image: ndarray, out: ndarray):
    gaussians = _initial_gaussians(blurred_image, watershedded_image)
    out.fill(0)
    for gaussian in gaussians:
        gaussian.add_to_image(out)


def _initial_gaussians(blurred_image: ndarray, watershedded_image: ndarray,) -> List[Gaussian]:
    gaussians = []
    centers = mahotas.center_of_mass(blurred_image, watershedded_image)
    for i in range(1, centers.shape[0]):
        center = centers[i]
        if numpy.isnan(center[0]):
            continue
        x, y, z = center[2], center[1], center[0]
        max = blurred_image[int(z), int(y), int(x)]
        gaussians.append(Gaussian(max, x, y, z, cov_xx=10, cov_yy=10, cov_zz=2,
                                  cov_xy=0, cov_xz=0, cov_yz=0))
    return gaussians


def gaussian_3d(positions, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    """Gets the value of the Gaussian at the given position.
    positions: [[z,y,x],[z,y,x],[z,y,x],...]
    a: prefactor, represents the intensity at mu
    mu: center position (or mean of a Gaussian distribution)
    cov_xx, cov_yy, cov_zz: widths (or standard deviations of a Gaussian distribution)
    cov_xy, cov_xz, cov_yz: skew in the given directions

    All the cov variables together form a covariance matrix.
    """
    mu = numpy.array([mu_x, mu_y, mu_z])
    covariance_matrix = numpy.array([
        [cov_xx, cov_xy, cov_xz],
        [cov_xy, cov_yy, cov_yz],
        [cov_xz, cov_yz, cov_zz]
    ])
    covariance_matrix_inversed = numpy.linalg.inv(covariance_matrix)

    output = numpy.empty(positions.shape[0], dtype=numpy.uint8)
    for i in range(output.shape[0]):
        pos = positions[i][::-1]  # positions is z, y, x, so change that to x, y, z using ::-1
        output[i] = _gaussian_3d_single(pos, a, mu, covariance_matrix_inversed)
    return output

def _gaussian_3d_single(pos: ndarray, a: float, mu: ndarray, cov_inv: ndarray):
    return a * numpy.exp(-1/2 * ((pos - mu) @ cov_inv @ (pos - mu)))

#scipy.optimize.curve_fit(gaussian_3d)