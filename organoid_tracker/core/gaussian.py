import math
from typing import Optional, List, Tuple, Any

import numpy
from numpy import ndarray

from organoid_tracker.core.bounding_box import BoundingBox
from organoid_tracker.core.ellipse import Ellipse


def _eigsorted(cov):
    vals, vecs = numpy.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


class Gaussian:
    """A three-dimensional Gaussian function."""

    a: float
    mu_x: float
    mu_y: float
    mu_z: float
    cov_xx: float
    cov_yy: float
    cov_zz: float
    cov_xy: float
    cov_xz: float
    cov_yz: float

    def __init__(self, a: float, mu_x: float, mu_y: float, mu_z: float, cov_xx: float, cov_yy: float, cov_zz: float,
                 cov_xy: float, cov_xz: float, cov_yz: float):
        if cov_xx < 0 or cov_yy < 0 or cov_zz < 0:
            raise ValueError(f"Standard deviations cannot be negative."
                             f" cov_xx={cov_xx}, cov_yy={cov_yy}, cov_zz={cov_zz}")

        self.a = a
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.mu_z = mu_z
        self.cov_xx = cov_xx
        self.cov_yy = cov_yy
        self.cov_zz = cov_zz
        self.cov_xy = cov_xy
        self.cov_xz = cov_xz
        self.cov_yz = cov_yz

    def draw(self, image: ndarray, cached_gaussian: Optional[ndarray] = None) -> Optional[ndarray]:
        """Draws a Gaussian to an image. Returns an array that can be passed again to this method (for these Gaussian
         parameters) to quickly redraw the Gaussian."""
        return self._draw_anything(image, _3d_gauss, cached_gaussian)

    def draw_ellipsoid(self, image: ndarray, cached_gaussian: Optional[ndarray] = None) -> Optional[ndarray]:
        """Draws a 3d ellipsoid to the given image. The image can be a boolean image to construct a mask."""
        return self._draw_anything(image, _3d_ellipsoid, cached_gaussian)

    def get_bounds(self) -> BoundingBox:
        """Gets the bounding box of this Gaussian function as min_x,min_y,min_z, max_x,max_y,max_z."""
        return BoundingBox(
            int(self.mu_x - 3 * math.sqrt(self.cov_xx)),
            int(self.mu_y - 3 * math.sqrt(self.cov_yy)),
            int(self.mu_z - 3 * math.sqrt(self.cov_zz)),

            int(self.mu_x + 3 * math.sqrt(self.cov_xx)),
            int(self.mu_y + 3 * math.sqrt(self.cov_yy)),
            int(self.mu_z + 3 * math.sqrt(self.cov_zz))
        )

    def draw_colored(self, image: ndarray, color: Tuple[float, float, float]) -> Optional[ndarray]:
        """Draws a Gaussian to an image in the given color. The color must be an RGB color, with numbers ranging from
        0 to 1. The Gaussian intensity is divided by 256, which should bring the Gaussian from the byte range [0..256]
        into the standard float range [0...1]."""
        if self.cov_xx < 0 or self.cov_yy < 0 or self.cov_zz < 0:
            return

        bounds = self.get_bounds()
        offset_x = max(0, bounds.min_x)
        offset_y = max(0, bounds.min_y)
        offset_z = max(0, bounds.min_z)
        max_x = min(image.shape[2], bounds.max_x)
        max_y = min(image.shape[1], bounds.max_y)
        max_z = min(image.shape[0], bounds.max_z)
        size_x, size_y, size_z = max_x - offset_x, max_y - offset_y, max_z - offset_z
        if size_x < 0 or size_y < 0 or size_z < 0:
            return

        pos = _get_positions(size_x, size_y, size_z)
        gauss = _3d_gauss(pos, self.a / 256, self.mu_x - offset_x, self.mu_y - offset_y, self.mu_z - offset_z,
                          self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy, self.cov_xz, self.cov_yz)
        try:
            cached_gaussian = gauss.reshape(size_z, size_y, size_x)
        except ValueError as e:
            raise e
        if color[0] > 0:
            image[offset_z:max_z, offset_y:max_y, offset_x:max_x, 0] += cached_gaussian * color[0]
        if color[1] > 0:
            image[offset_z:max_z, offset_y:max_y, offset_x:max_x, 1] += cached_gaussian * color[1]
        if color[2] > 0:
            image[offset_z:max_z, offset_y:max_y, offset_x:max_x, 2] += cached_gaussian * color[2]
        return cached_gaussian

    def _draw_anything(self, image: ndarray, draw_function, cached_result: Optional[ndarray] = None):
        if self.cov_xx < 0 or self.cov_yy < 0 or self.cov_zz < 0 \
                or self.mu_x < 0 or self.mu_x > image.shape[2] \
                or self.mu_y < 0 or self.mu_y > image.shape[1] \
                or self.mu_z < 0 or self.mu_z > image.shape[0]:
            return  # All invalid Gaussians

        bounds = self.get_bounds()
        offset_x = max(0, bounds.min_x)
        offset_y = max(0, bounds.min_y)
        offset_z = max(0, bounds.min_z)
        max_x = min(image.shape[2], bounds.max_x)
        max_y = min(image.shape[1], bounds.max_y)
        max_z = min(image.shape[0], bounds.max_z)

        if cached_result is None:
            # Need to calculate
            size_x, size_y, size_z = max_x - offset_x, max_y - offset_y, max_z - offset_z
            pos = _get_positions(size_x, size_y, size_z)
            cached_result = draw_function(pos, self.a, self.mu_x - offset_x, self.mu_y - offset_y, self.mu_z - offset_z,
                                          self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy, self.cov_xz, self.cov_yz)
            cached_result = cached_result.reshape(size_z, size_y, size_x)
        image[offset_z:max_z, offset_y:max_y, offset_x:max_x] += cached_result
        return cached_result

    def draw_gradient(self, image: ndarray, gradient_nr: int):
        """
        Gets the Nth gradient for all pixels in an image of the given size. Gradient nr. 0 is dG/da, nr. 1 is dG/dmu_x,
        etc. Order is the same as in Gaussian(...) and self.to_list()
        """
        self._draw_anything(image, _GRADIENT_FUNCTIONS[gradient_nr])

    def to_list(self) -> List[float]:
        return [self.a, self.mu_x, self.mu_y, self.mu_z, self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy,
                self.cov_xz, self.cov_yz]

    def to_ellipse(self, number_of_standards_devs: float = 1.5) -> Ellipse:
        """Gets an ellipse that describes the 2D shape of this Gaussian at z = 0. The higher number_of_standard_devs,
        the larger the ellipse."""
        cov = numpy.array([[self.cov_xx, self.cov_xy],
                           [self.cov_xy, self.cov_yy]])

        vals, vecs = _eigsorted(cov)
        angle: float = numpy.degrees(numpy.arctan2(*vecs[:, 0][::-1]))
        width, height = number_of_standards_devs * 2 * numpy.sqrt(vals)
        if height < width:
            height, width = width, height
            angle = (angle + 90) % 180
        return Ellipse(x=self.mu_x, y=self.mu_y, width=width, height=height, angle=angle)

    def almost_equal(self, other: "Gaussian", a_delta=10, mu_delta=1, cov_delta=2) -> bool:
        return abs(self.a - other.a) < a_delta and \
               abs(self.mu_x - other.mu_x) < mu_delta and \
               abs(self.mu_y - other.mu_y) < mu_delta and \
               abs(self.mu_z - other.mu_z) < mu_delta and \
               abs(self.cov_xx - other.cov_xx) < cov_delta and \
               abs(self.cov_yy - other.cov_yy) < cov_delta and \
               abs(self.cov_zz - other.cov_zz) < cov_delta and \
               abs(self.cov_xy - other.cov_xy) < cov_delta and \
               abs(self.cov_xz - other.cov_xz) < cov_delta and \
               abs(self.cov_yz - other.cov_yz) < cov_delta

    def translated(self, dx: float, dy: float, dz: float) -> "Gaussian":
        new_gaussian = Gaussian(*self.to_list())
        new_gaussian.mu_x += dx
        new_gaussian.mu_y += dy
        new_gaussian.mu_z += dz
        return new_gaussian

    def __eq__(self, other: Any) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self) -> int:
        return hash((self.a, self.mu_x, self.mu_y, self.mu_z, self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy,
                     self.cov_xz, self.cov_yz))

    def __repr__(self) -> str:
        return f"Gaussian(a={self.a:.2f}, mu_x={self.mu_x:.2f}, mu_y={self.mu_y:.2f}, mu_z={self.mu_z:.2f}," \
               f" cov_xx={self.cov_xx:.2f}, cov_yy={self.cov_yy:.2f}, cov_zz={self.cov_zz:.2f}," \
               f" cov_xy={self.cov_xy:.2f}, cov_xz={self.cov_xz:.2f}, cov_yz={self.cov_yz:.2f})"


def _get_positions(xsize: int, ysize: int, zsize: int) -> ndarray:
    """Creates a list of x/y/z positions: [[x1,y1,z1],[x2,y2,z2],...]. Every possible position in the image is returned.
    """
    x = numpy.arange(xsize)
    y = numpy.arange(ysize)
    z = numpy.arange(zsize)
    y, z, x = numpy.meshgrid(y, z, x)
    return numpy.column_stack([x.ravel(), y.ravel(), z.ravel()])


def _3d_gauss(pos: ndarray, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz) -> ndarray:
    """Calculates a 3D Gaussian for the given positions.
    :param pos: Stack of vectors: [[x1, y1, z1], [x2, y2, z2], ...] (so pos[0] is the first position). Can also be a
    single vector: [x, y, z].
    :param a: Intensity at mean position.
    :param mu_x: X of mean position.
    :param cov_xx: Entry in covariance matrix.
    :return: Gaussian intensities for all given vectors: [I1, I2, ...]
    """
    pos = pos[..., numpy.newaxis]  # From list of vectors to list of column vectors
    mu = numpy.array([[mu_x], [mu_y], [mu_z]])  # A column vector
    covariance_matrix = numpy.array([
        [cov_xx, cov_xy, cov_xz],
        [cov_xy, cov_yy, cov_yz],
        [cov_xz, cov_yz, cov_zz]
    ])
    cov_inv = numpy.linalg.inv(covariance_matrix)

    pos_mu = pos - mu
    transpose_axes = (0, 2, 1) if len(pos_mu.shape) == 3 else (1, 0)
    pos_mu_T = numpy.transpose(pos_mu, transpose_axes)

    return a * numpy.exp(-1 / 2 * (pos_mu_T @ cov_inv @ pos_mu).ravel())


def _3d_ellipsoid(pos: ndarray, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz,
                  cutoff=0.2) -> ndarray:
    """Returns an ellipse with roughly the same shape as a Gaussian. All places where the intensity >= 20% of the
    maximum are set to True, the others to False.
    """
    pos = pos[..., numpy.newaxis]  # From list of vectors to list of column vectors
    mu = numpy.array([[mu_x], [mu_y], [mu_z]])  # A column vector
    covariance_matrix = numpy.array([
        [cov_xx, cov_xy, cov_xz],
        [cov_xy, cov_yy, cov_yz],
        [cov_xz, cov_yz, cov_zz]
    ])
    cov_inv = numpy.linalg.inv(covariance_matrix)

    pos_mu = pos - mu
    transpose_axes = (0, 2, 1) if len(pos_mu.shape) == 3 else (1, 0)
    pos_mu_T = numpy.transpose(pos_mu, transpose_axes)

    return (pos_mu_T @ cov_inv @ pos_mu).ravel() < -2 * numpy.log(cutoff)


# Partial derivatives of the above function
def _deriv_a(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    """Returns the partial derivative dG/da"""
    return _3d_gauss(pos, 1, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz)


# Autogenerated by a Mathematica script
def _deriv_mu_x(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return -((a * (cov_yz ** 2 * (mu_x - pos_x) + cov_yy * cov_zz * (-mu_x + pos_x) + cov_xy * cov_zz * (
                mu_y - pos_y) + cov_xz * cov_yz * (-mu_y + pos_y) + cov_xz * cov_yy * (
                               mu_z - pos_z) + cov_xy * cov_yz * (-mu_z + pos_z))) / ((
                                                                                                  cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                                      cov_yz ** 2 - cov_yy * cov_zz)) * numpy.exp(
        (
                    2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                        mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                cov_xz * (mu_x - pos_x) * (mu_y - pos_y) + (
                                    cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                            mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                cov_zz * (mu_x - pos_x) ** 2 + (mu_z - pos_z) * (
                                    -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (2 * (
                    cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                        cov_yz ** 2 - cov_yy * cov_zz))))))


def _deriv_mu_y(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return -((a * (cov_xy * cov_zz * (mu_x - pos_x) + cov_xz * cov_yz * (-mu_x + pos_x) + cov_xz ** 2 * (
                mu_y - pos_y) + cov_xx * cov_zz * (-mu_y + pos_y) + cov_xx * cov_yz * (
                               mu_z - pos_z) + cov_xy * cov_xz * (-mu_z + pos_z))) / ((
                                                                                                  cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                                      cov_yz ** 2 - cov_yy * cov_zz)) * numpy.exp(
        (
                    2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                        mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                cov_xz * (mu_x - pos_x) * (mu_y - pos_y) + (
                                    cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                            mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                cov_zz * (mu_x - pos_x) ** 2 + (mu_z - pos_z) * (
                                    -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (2 * (
                    cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                        cov_yz ** 2 - cov_yy * cov_zz))))))


def _deriv_mu_z(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return -((a * (cov_xz * cov_yy * (mu_x - pos_x) + cov_xy * cov_yz * (-mu_x + pos_x) + cov_xx * cov_yz * (
                mu_y - pos_y) + cov_xy * cov_xz * (-mu_y + pos_y) + cov_xy ** 2 * (mu_z - pos_z) + cov_xx * cov_yy * (
                               -mu_z + pos_z))) / ((
                                                               cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                   cov_yz ** 2 - cov_yy * cov_zz)) * numpy.exp((
                                                                                                                           2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                                                                                                                               mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                                                                                                                       cov_xz * (
                                                                                                                                           mu_x - pos_x) * (
                                                                                                                                                   mu_y - pos_y) + (
                                                                                                                                                   cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                                                                                                                                   mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                                                                                                                       cov_zz * (
                                                                                                                                           mu_x - pos_x) ** 2 + (
                                                                                                                                                   mu_z - pos_z) * (
                                                                                                                                                   -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (
                                                                                                                           2 * (
                                                                                                                               cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                                                                   cov_yz ** 2 - cov_yy * cov_zz))))))


def _deriv_cov_xx(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return (a * (cov_yz ** 2 * (mu_x - pos_x) + cov_yy * cov_zz * (-mu_x + pos_x) + cov_xy * cov_zz * (
                mu_y - pos_y) + cov_xz * cov_yz * (-mu_y + pos_y) + cov_xz * cov_yy * (
                             mu_z - pos_z) + cov_xy * cov_yz * (-mu_z + pos_z)) ** 2) / (2 * (
                cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                    cov_yz ** 2 - cov_yy * cov_zz)) ** 2 * numpy.exp((
                                                                                 2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                                                                                     mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                                                                             cov_xz * (mu_x - pos_x) * (
                                                                                                 mu_y - pos_y) + (
                                                                                                         cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                                                                                         mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                                                                             cov_zz * (
                                                                                                 mu_x - pos_x) ** 2 + (
                                                                                                         mu_z - pos_z) * (
                                                                                                         -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (
                                                                                 2 * (
                                                                                     cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                         cov_yz ** 2 - cov_yy * cov_zz)))))


def _deriv_cov_yy(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return (a * (cov_xy * cov_zz * (mu_x - pos_x) + cov_xz * cov_yz * (-mu_x + pos_x) + cov_xz ** 2 * (
                mu_y - pos_y) + cov_xx * cov_zz * (-mu_y + pos_y) + cov_xx * cov_yz * (
                             mu_z - pos_z) + cov_xy * cov_xz * (-mu_z + pos_z)) ** 2) / (2 * (
                cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                    cov_yz ** 2 - cov_yy * cov_zz)) ** 2 * numpy.exp((
                                                                                 2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                                                                                     mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                                                                             cov_xz * (mu_x - pos_x) * (
                                                                                                 mu_y - pos_y) + (
                                                                                                         cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                                                                                         mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                                                                             cov_zz * (
                                                                                                 mu_x - pos_x) ** 2 + (
                                                                                                         mu_z - pos_z) * (
                                                                                                         -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (
                                                                                 2 * (
                                                                                     cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                         cov_yz ** 2 - cov_yy * cov_zz)))))


def _deriv_cov_zz(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return (a * (cov_xz * cov_yy * (mu_x - pos_x) + cov_xy * cov_yz * (-mu_x + pos_x) + cov_xx * cov_yz * (
                mu_y - pos_y) + cov_xy * cov_xz * (-mu_y + pos_y) + cov_xy ** 2 * (mu_z - pos_z) + cov_xx * cov_yy * (
                             -mu_z + pos_z)) ** 2) / (2 * (
                cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                    cov_yz ** 2 - cov_yy * cov_zz)) ** 2 * numpy.exp((
                                                                                 2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                                                                                     mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                                                                             cov_xz * (mu_x - pos_x) * (
                                                                                                 mu_y - pos_y) + (
                                                                                                         cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                                                                                         mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                                                                             cov_zz * (
                                                                                                 mu_x - pos_x) ** 2 + (
                                                                                                         mu_z - pos_z) * (
                                                                                                         -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (
                                                                                 2 * (
                                                                                     cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                         cov_yz ** 2 - cov_yy * cov_zz)))))


def _deriv_cov_xy(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return -((a * (cov_xy * cov_zz * (mu_x - pos_x) + cov_xz * cov_yz * (-mu_x + pos_x) + cov_xz ** 2 * (
                mu_y - pos_y) + cov_xx * cov_zz * (-mu_y + pos_y) + cov_xx * cov_yz * (
                               mu_z - pos_z) + cov_xy * cov_xz * (-mu_z + pos_z)) * (
                          cov_yy * cov_zz * (mu_x - pos_x) + cov_yz ** 2 * (-mu_x + pos_x) + cov_xz * cov_yz * (
                              mu_y - pos_y) + cov_xy * cov_zz * (-mu_y + pos_y) + cov_xy * cov_yz * (
                                      mu_z - pos_z) + cov_xz * cov_yy * (-mu_z + pos_z))) / ((
                                                                                                         cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                                             cov_yz ** 2 - cov_yy * cov_zz)) ** 2 * numpy.exp(
        (
                    2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                        mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                cov_xz * (mu_x - pos_x) * (mu_y - pos_y) + (
                                    cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                            mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                cov_zz * (mu_x - pos_x) ** 2 + (mu_z - pos_z) * (
                                    -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (2 * (
                    cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                        cov_yz ** 2 - cov_yy * cov_zz))))))


def _deriv_cov_xz(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return (a * (cov_xz * cov_yy * (mu_x - pos_x) + cov_xy * cov_yz * (-mu_x + pos_x) + cov_xx * cov_yz * (
                mu_y - pos_y) + cov_xy * cov_xz * (-mu_y + pos_y) + cov_xy ** 2 * (mu_z - pos_z) + cov_xx * cov_yy * (
                             -mu_z + pos_z)) * (
                        cov_yz ** 2 * (mu_x - pos_x) + cov_yy * cov_zz * (-mu_x + pos_x) + cov_xy * cov_zz * (
                            mu_y - pos_y) + cov_xz * cov_yz * (-mu_y + pos_y) + cov_xz * cov_yy * (
                                    mu_z - pos_z) + cov_xy * cov_yz * (-mu_z + pos_z))) / ((
                                                                                                       cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                                           cov_yz ** 2 - cov_yy * cov_zz)) ** 2 * numpy.exp(
        (
                    2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                        mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                cov_xz * (mu_x - pos_x) * (mu_y - pos_y) + (
                                    cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                            mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                cov_zz * (mu_x - pos_x) ** 2 + (mu_z - pos_z) * (
                                    -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (2 * (
                    cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                        cov_yz ** 2 - cov_yy * cov_zz)))))


def _deriv_cov_yz(pos, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
    pos_x, pos_y, pos_z = pos[:, 0], pos[:, 1], pos[:, 2]
    return (a * (cov_xy * cov_zz * (mu_x - pos_x) + cov_xz * cov_yz * (-mu_x + pos_x) + cov_xz ** 2 * (
                mu_y - pos_y) + cov_xx * cov_zz * (-mu_y + pos_y) + cov_xx * cov_yz * (
                             mu_z - pos_z) + cov_xy * cov_xz * (-mu_z + pos_z)) * (
                        cov_xz * cov_yy * (mu_x - pos_x) + cov_xy * cov_yz * (-mu_x + pos_x) + cov_xx * cov_yz * (
                            mu_y - pos_y) + cov_xy * cov_xz * (-mu_y + pos_y) + cov_xy ** 2 * (
                                    mu_z - pos_z) + cov_xx * cov_yy * (-mu_z + pos_z))) / ((
                                                                                                       cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                                                                                                           cov_yz ** 2 - cov_yy * cov_zz)) ** 2 * numpy.exp(
        (
                    2 * cov_xy * cov_zz * mu_x * mu_y + cov_xz ** 2 * mu_y ** 2 - cov_xx * cov_zz * mu_y ** 2 - 2 * cov_xy * cov_xz * mu_y * mu_z + cov_xy ** 2 * mu_z ** 2 + cov_yz ** 2 * (
                        mu_x - pos_x) ** 2 - 2 * cov_xy * cov_zz * mu_y * pos_x - 2 * cov_xy * cov_zz * mu_x * pos_y - 2 * cov_xz ** 2 * mu_y * pos_y + 2 * cov_xx * cov_zz * mu_y * pos_y + 2 * cov_xy * cov_xz * mu_z * pos_y + 2 * cov_xy * cov_zz * pos_x * pos_y + cov_xz ** 2 * pos_y ** 2 - cov_xx * cov_zz * pos_y ** 2 - 2 * cov_yz * (
                                cov_xz * (mu_x - pos_x) * (mu_y - pos_y) + (
                                    cov_xy * mu_x - cov_xx * mu_y - cov_xy * pos_x + cov_xx * pos_y) * (
                                            mu_z - pos_z)) + 2 * cov_xy * cov_xz * mu_y * pos_z - 2 * cov_xy ** 2 * mu_z * pos_z - 2 * cov_xy * cov_xz * pos_y * pos_z + cov_xy ** 2 * pos_z ** 2 - cov_yy * (
                                cov_zz * (mu_x - pos_x) ** 2 + (mu_z - pos_z) * (
                                    -2 * cov_xz * mu_x + cov_xx * mu_z + 2 * cov_xz * pos_x - cov_xx * pos_z))) / (2 * (
                    cov_xz ** 2 * cov_yy - 2 * cov_xy * cov_xz * cov_yz + cov_xy ** 2 * cov_zz + cov_xx * (
                        cov_yz ** 2 - cov_yy * cov_zz)))))


_GRADIENT_FUNCTIONS = [  # The gradient functions in the same parameter order as Gaussian(...) and gaussian.to_list()
    _deriv_a,
    _deriv_mu_x, _deriv_mu_y, _deriv_mu_z,
    _deriv_cov_xx, _deriv_cov_yy, _deriv_cov_zz,
    _deriv_cov_xy, _deriv_cov_xz, _deriv_cov_yz
]
