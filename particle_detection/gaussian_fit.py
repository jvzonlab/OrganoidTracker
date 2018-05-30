import cv2
from typing import Tuple, List, Iterable, Generator, Set

import networkx
from numpy import ndarray
import scipy.optimize
import numpy
from tifffile import tifffile

from core import Particle
from networkx import Graph

from particle_detection import watershedding
from particle_detection.ellipse import Ellipse, EllipseStack, EllipseCluster
import matplotlib.pyplot as plt


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

    def __init__(self, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
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

    def draw(self, image: ndarray):
        offset_x = max(0, int(self.mu_x - 3 * self.cov_xx))
        offset_y = max(0, int(self.mu_y - 3 * self.cov_yy))
        offset_z = max(0, int(self.mu_z - 3 * self.cov_zz))
        max_x = min(image.shape[2], int(self.mu_x + 3 * self.cov_xx))
        max_y = min(image.shape[1], int(self.mu_y + 3 * self.cov_yy))
        max_z = min(image.shape[0], int(self.mu_z + 3 * self.cov_zz))
        size_x, size_y, size_z = max_x - offset_x, max_y - offset_y, max_z - offset_z

        pos = _get_positions(size_x, size_y, size_z)
        gauss = _3d_gauss(pos, self.a, self.mu_x - offset_x, self.mu_y - offset_y, self.mu_z - offset_z,
                          self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy, self.cov_xz, self.cov_yz)
        gauss = gauss.reshape(size_z, size_y, size_x)
        image[offset_z:max_z, offset_y:max_y, offset_x:max_x] += gauss

    def to_list(self) -> List[float]:
        return [self.a, self.mu_x, self.mu_y, self.mu_z, self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy,
                self.cov_xz, self.cov_yz]

    def almost_equal(self, other: "Gaussian", a_delta=10, mu_delta=1, cov_delta=1) -> bool:
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

    def scaled(self, scale: float):
        new_gaussian = Gaussian(*self.to_list())
        new_gaussian.mu_x *= scale
        new_gaussian.mu_y *= scale
        new_gaussian.mu_z *= scale
        # Todo: resize covariance matrix
        return new_gaussian

    def __repr__(self):
        return "Gaussian(*" + repr(self.to_list()) + ")"

def particles_to_gaussians(image: ndarray, particles: Iterable[Particle]) -> List[Gaussian]:
    gaussians = []
    for particle in particles:
        intensity = image[int(particle.z), int(particle.y), int(particle.x)]
        gaussians.append(Gaussian(intensity, particle.x, particle.y, particle.z, 15, 15, 3, 0, 0, 0))
    return gaussians


def _3d_gauss_multiple(pos: ndarray, *args):
    """Gaussian mixture model. Every Gaussian has 10 parameters: the first 10 args go to the first Gaussian (see
     _3d_gauss), argument 10 - 19 to the second, etc."""
    args_count = len(args)
    totals = numpy.zeros(pos.shape[0], dtype=numpy.float32)
    for i in range(0, args_count, 10):
        gaussian_params = args[i:i + 10]
        totals += _3d_gauss(pos, *gaussian_params)
    return totals


def _3d_gauss(pos: ndarray, a, mu_x, mu_y, mu_z, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz) -> ndarray:
    """Calculates a 3D Gaussian for the given positions.
    :param pos: Stack of vectors: [[x1, y1, z1], [x2, y2, z2], ...]. Can also be a single vector: [x, y, z].
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


def _get_positions(xsize: int, ysize: int, zsize: int) -> ndarray:
    """Creates a list of x/y/z positions: [[x1,y1,z1],[x2,y2,z2],...]. The order of the positions is such that these
    represent the x,y,z coords of the elements of zyx_array.ravel()."""
    x = numpy.arange(xsize)
    y = numpy.arange(ysize)
    z = numpy.arange(zsize)
    y, z, x = numpy.meshgrid(y, z, x)
    return numpy.column_stack([x.ravel(), y.ravel(), z.ravel()])


def add_noise(data: ndarray):
    """Adds noise to the given data. Useful for construction of artificial testing data."""
    shape = data.shape
    numpy.random.seed(1949)  # Make sure noise is reproducible
    data = data.ravel()
    data += 20 * numpy.random.normal(size=len(data))
    return data.reshape(*shape)


def perform_gaussian_fit(original_image: ndarray, guess: Gaussian) -> Gaussian:
    """Fits a gaussian function to an image. original_image is a zyx-indexed image, guess is an initial starting point
    for the fit."""
    return perform_gaussian_mixture_fit(original_image, [guess])[0]


def perform_gaussian_mixture_fit(original_image: ndarray, guesses: Iterable[Gaussian]) -> List[Gaussian]:
    """Fits multiple Gaussians to the image (a Gaussian Mixture Model). Initial seeds must be given."""
    xsize, ysize, zsize = original_image.shape[2], original_image.shape[1], original_image.shape[0]
    pos = _get_positions(xsize, ysize, zsize)
    parameters = []
    for guess in guesses:
        parameters += guess.to_list()

    image_1d = original_image.ravel()
    print("Starting the fit... " + str(len(parameters)) + " parameters, " + str(pos.shape[0]) + " pixels")
    coeff, var_matrix = scipy.optimize.curve_fit(_3d_gauss_multiple, pos, image_1d, p0=parameters, maxfev=10000)
    print("Done!")
    gaussians = []
    for i in range(0, len(coeff), 10):
        gaussians.append(Gaussian(*coeff[i:i + 10]))
    return gaussians


def perform_gaussian_mixture_fit_from_watershed(watershed_image: ndarray, out: ndarray):
    """GMM using watershed as seeds. out is a color image where the detected Gaussians can be drawn on."""
    ellipse_stacks = _get_ellipse_stacks(watershed_image)
    ellipse_stacks = _get_overlapping_stacks(ellipse_stacks)

    i = 1
    for stacks in ellipse_stacks:
        color = watershedding.COLOR_ARRAY[i]
        color_opencv = color[0] * 255, color[1] * 255, color[2] * 255

        stacks.draw_to_image(out, color_opencv)
        i += 1


def _get_ellipse_stacks(watershed: ndarray) -> List[EllipseStack]:
    max = watershed.max()
    buffer = numpy.empty_like(watershed, dtype=numpy.uint8)
    ellipse_stacks = []
    for i in range(1, max):
        ellipse_stack = []
        buffer.fill(0)
        buffer[watershed == i] = 255
        for z in range(buffer.shape[0]):
            contour_image, contours, hierarchy = cv2.findContours(buffer[z], cv2.RETR_LIST, 2)
            contour_index, area = _find_contour_with_largest_area(contours)
            if contour_index == -1 or area < 40:
                ellipse_stack.append(None)
                continue  # No contours found
            ellipse_pos, ellipse_size, ellipse_angle = cv2.fitEllipse(contours[contour_index])
            ellipse_stack.append(Ellipse(ellipse_pos[0], ellipse_pos[1], ellipse_size[0] - 2, ellipse_size[1] - 2, ellipse_angle))
        ellipse_stacks.append(EllipseStack(ellipse_stack))
    return ellipse_stacks


def _get_overlapping_stacks(stacks: List[EllipseStack]) -> List[EllipseCluster]:
    cell_network = Graph()
    for stack in stacks:
        cell_network.add_node(stack)
        for other_stack in stacks:
            if other_stack is stack:
                continue  # Ignore self-overlapping
            if other_stack not in cell_network:
                continue  # To be processed later
            if stack.intersects(other_stack):
                cell_network.add_edge(stack, other_stack)

    clusters = []
    for cluster in networkx.connected_components(cell_network):
        clusters.append(EllipseCluster(cluster))
    return clusters

def _find_contour_with_largest_area(contours) -> Tuple[int, float]:
    highest_area = 0
    index_with_highest_area = -1
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > highest_area:
            highest_area = area
            index_with_highest_area = i
    return index_with_highest_area, highest_area
