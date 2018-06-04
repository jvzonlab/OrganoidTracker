from timeit import default_timer as timer
from typing import Tuple, List, Iterable, Dict, Optional

import cv2
import networkx
import numpy
import scipy.optimize
from networkx import Graph
from numpy import ndarray

from core import Particle
from particle_detection import watershedding, smoothing
from particle_detection.ellipse import Ellipse, EllipseStack, EllipseCluster


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

    def draw(self, image: ndarray, cached_gaussian: Optional[ndarray] = None):
        """Draws a Gaussian to an image. Returns an array that can be passed again to this method (for these Gaussian
         parameters) to quickly redraw the Gaussian."""
        offset_x = max(0, int(self.mu_x - 3 * self.cov_xx))
        offset_y = max(0, int(self.mu_y - 3 * self.cov_yy))
        offset_z = max(0, int(self.mu_z - 3 * self.cov_zz))
        max_x = min(image.shape[2], int(self.mu_x + 3 * self.cov_xx))
        max_y = min(image.shape[1], int(self.mu_y + 3 * self.cov_yy))
        max_z = min(image.shape[0], int(self.mu_z + 3 * self.cov_zz))

        if cached_gaussian is None:
            size_x, size_y, size_z = max_x - offset_x, max_y - offset_y, max_z - offset_z
            pos = _get_positions(size_x, size_y, size_z)
            gauss = _3d_gauss(pos, self.a, self.mu_x - offset_x, self.mu_y - offset_y, self.mu_z - offset_z,
                              self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy, self.cov_xz, self.cov_yz)
            cached_gaussian = gauss.reshape(size_z, size_y, size_x)
        image[offset_z:max_z, offset_y:max_y, offset_x:max_x] += cached_gaussian
        return cached_gaussian

    def to_list(self) -> List[float]:
        return [self.a, self.mu_x, self.mu_y, self.mu_z, self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy,
                self.cov_xz, self.cov_yz]

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

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self):
        return hash((self.a, self.mu_x, self.mu_y, self.mu_z, self.cov_xx, self.cov_yy, self.cov_zz, self.cov_xy,
                self.cov_xz, self.cov_yz))

    def __repr__(self):
        return "Gaussian(*" + repr(self.to_list()) + ")"


def particles_to_gaussians(image: ndarray, particles: Iterable[Particle]) -> List[Gaussian]:
    gaussians = []
    for particle in particles:
        intensity = image[int(particle.z), int(particle.y), int(particle.x)]
        gaussians.append(Gaussian(intensity, particle.x, particle.y, particle.z, 15, 15, 3, 0, 0, 0))
    return gaussians


class _ModelAndImageDifference:
    _data_image: ndarray
    _scratch_image: ndarray
    _last_gaussians: Dict[Gaussian, ndarray]

    def __init__(self, data_image: ndarray):
        self._data_image = data_image.astype(numpy.float64)
        self._scratch_image = numpy.empty_like(self._data_image)
        self._last_gaussians = dict()

    def difference_with_image(self, params) -> float:
        last_gaussians_new = dict()

        self._scratch_image.fill(0)
        for i in range(0, len(params), 10):
            gaussian_params = params[i:i + 10]
            gaussian = Gaussian(*gaussian_params)
            cached_image = self._last_gaussians.get(gaussian)
            last_gaussians_new[gaussian] = gaussian.draw(self._scratch_image, cached_image)
        self._last_gaussians = last_gaussians_new

        self._scratch_image -= self._data_image
        self._scratch_image **= 2
        sum = self._scratch_image.sum()
        print("Difference: " +  '{0:.16f}'.format(sum) + ". Params: " + str(params))
        return sum


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
    model_and_image_difference = _ModelAndImageDifference(original_image)

    # guesses_list = []
    # for guess in guesses:
    #     guesses_list += guess.to_list()
    result_x = [ -1.75240029e+00,  3.49333980e+02, -2.77228338e+03,  9.48127663e+00
,  8.18543208e+03,  8.44411317e+10,  6.34363406e+01, -2.25041952e+07
,  5.08036063e+01, -3.21129165e+05,  3.84526823e+02,  4.02950733e+02
,  1.26053890e+02,  7.99634344e+00,  3.54169293e+01,  5.69875032e+01
,  3.46678038e+00,  2.89123238e+00,  2.29722016e+00, -3.62836368e+00
,  3.45192812e+02,  3.79755452e+02,  1.07816738e+02,  7.74467396e+00
,  7.16511705e+01,  6.07696165e+01,  2.48289683e+00, -1.49236178e+01
,  3.28024040e-01, -3.69090790e+00,  2.94071761e+02,  3.41302064e+02
,  2.15128745e+02,  1.77991034e+01,  6.37836729e+01,  7.57535222e+01
,  1.35069791e+01, -2.60078073e+01,  3.46616592e+00, -6.12644389e+00
,  4.13002242e+02,  3.67682547e+02,  1.31843730e+02,  5.17693040e+00
,  6.22744906e+01,  6.16731286e+01,  2.56788807e+00,  3.24368688e+00
,  1.38855524e+00, -9.07498370e-01,  3.86793192e+02,  3.54645393e+02
,  1.03349384e+02,  8.09818772e+00,  8.37200850e+01,  4.19500568e+01
,  1.98161881e+00, -2.15681939e+01, -1.36436420e+00, -1.14282939e-01
,  2.96245546e+02,  3.23161074e+02,  1.10324830e+02,  8.50866509e+00
,  8.77229561e+01,  5.96149746e+01,  3.00679549e+00,  2.41407670e+00
, -4.76468420e-01, -3.94988500e+00,  2.24734743e+02,  3.94836639e+02
,  1.09325824e+02,  1.24995629e+01,  1.20762138e+02,  8.13077554e+01
,  3.27357156e+00,  4.55062458e+01,  3.77298369e+00,  3.03744595e+00
,  3.34641992e+02,  4.14193782e+02,  1.46204574e+02,  1.14381640e+01
,  5.68713514e+01,  9.44062408e+01,  2.51231960e+00,  2.83277361e+00
, -2.09245331e+00, -1.29848602e+00,  2.95473906e+02,  4.05417832e+02
,  1.72042801e+02,  1.13817960e+01,  5.31758808e+01,  6.54751786e+01
,  2.47636187e+00,  9.99088917e+00, -3.65308615e-02, -4.52771977e+00
,  3.05827467e+02,  3.85775262e+02,  1.37807484e+02,  1.05558078e+01
,  4.13751058e+01,  2.98852078e+01,  1.44203607e+00,  2.83397282e+00
, -2.51333004e+00,  2.88519486e+00,  3.62550335e+02,  3.72996436e+02
,  1.59508453e+02,  8.46515748e+00,  5.65857118e+01,  1.66971254e+01
,  1.69273808e+00,  2.83363217e+00, -1.06860673e-01,  1.51079197e+00
,  3.74092876e+02,  3.86772136e+02,  1.67378832e+02,  6.05242252e+00
,  8.87886405e+01,  4.49331947e+01,  2.21039214e+00, -1.72580017e+01
,  8.57454633e+00, -2.69473400e+00,  3.37048650e+02,  3.91299661e+02
,  1.86883849e+02,  6.59618046e+00,  7.44527881e+01,  6.65396811e+01
,  1.80085617e+00, -2.09290473e+01,  3.45171156e+00, -4.07958415e+00
,  3.61227589e+02,  3.49175943e+02,  1.69322855e+02,  5.52619686e+00
,  1.06918649e+02,  8.19987398e+01,  1.49119971e+00,  1.76874518e+01
,  1.96106890e+00,  1.27335048e+00,  2.90852797e+02,  3.89540661e+02
,  1.94035102e+02,  9.54911415e+00,  6.61451309e+01,  5.96818596e+01
,  2.94098127e+00,  3.49664001e+00,  2.04902033e+00, -1.13709545e+00
,  1.83446954e+02,  3.44139467e+02,  2.01394516e+02,  9.22946431e+00
,  1.77333043e+02,  6.41130122e+01,  7.64615347e+00, -8.21867486e+01
, -1.04707141e+01,  6.94817514e+00,  3.27589745e+02,  3.58998614e+02
,  9.42846305e+01,  1.27617697e+01,  5.89217984e+01,  4.40781541e+01
,  2.75471932e+00,  5.85718270e+00,  5.51124206e+00,  1.37372916e+00
,  3.65431232e+02,  3.34446624e+02,  1.91927530e+02,  9.24842851e+00
,  7.83894403e+01,  6.64686545e+01,  7.16530706e+00, -4.68015157e+00
, -1.45012888e+00,  6.98279613e+00,  3.12472906e+02,  3.65404946e+02
,  2.02899130e+02,  9.08727245e+00,  3.62289137e+01,  6.93106798e+01
,  6.14836930e+00,  3.56555330e+00,  5.23492538e-01,  3.97350922e+00
,  3.31094782e+02,  3.23860119e+02,  1.32268246e+02,  6.40608445e+00
,  7.98815928e+01,  5.09280817e+01,  3.03019978e+00,  1.42890137e+01
, -2.60186349e+00, -4.17935600e+00,  3.77418959e+02,  3.14746737e+02
,  1.48232261e+02,  6.67079514e+00,  4.28135087e+01,  4.72323595e+01
,  2.49542913e+00,  4.43195357e+00,  1.97801588e+00, -5.45542239e+00
,  3.88728442e+02,  3.05744991e+02,  2.13713170e+02,  8.00288147e+00
,  2.69898032e+01,  1.05157091e+02,  2.82480854e+00, -4.87126275e+00
, -4.65018188e-01,  3.31164703e+00,  2.61690888e+02,  3.54058494e+02
,  2.04849655e+02,  1.44112016e+01,  5.75044977e+01,  9.19181228e+01
,  2.77529120e+00, -1.78515932e+01,  4.97879775e+00, -7.02180839e+00
,  2.16872138e+02,  3.87817691e+02,  1.93623978e+02,  1.30419061e+01
,  9.32991096e+01,  9.77367021e+01,  1.19755628e+00, -1.19995862e+01
,  6.83332366e-01, -3.76457070e+00,  1.37029049e+02,  3.72838159e+02
,  1.95035817e+02,  1.40974493e+01,  4.81228324e+01,  2.28300182e+01
,  2.58475496e+00,  9.91968550e-01,  8.29670367e+00, -2.20320233e+00
,  3.38288918e+02,  3.11430073e+02,  1.28280494e+02,  1.26267286e+01
,  4.49147580e+01,  4.35558825e+01,  2.71588061e+00, -2.40632060e+00
,  4.97147087e+00, -5.65197905e+00,  1.49926186e+02,  3.19053503e+02
,  1.08462426e+02,  1.77700610e+01,  6.10821492e+01,  4.48122294e+01
,  9.31358295e+00,  1.66316616e+01, -1.49974253e+01, -1.05198497e+01
,  1.89412682e+02,  3.38381525e+02,  1.01571334e+02,  1.43408444e+01
,  6.98841234e+01,  6.67065747e+01,  3.22782931e+00, -5.22034784e+00
,  5.23979239e+00,  9.69274704e-02,  1.87412827e+02,  3.91354975e+02
,  1.48109945e+02,  1.65895539e+01,  5.48314630e+01,  6.63679396e+01
,  2.41829319e+00,  3.44096601e+01,  7.12041153e-02, -2.25486225e+00
,  3.43336307e+02,  2.73414924e+02,  1.48557046e+02,  1.26546488e+01
,  2.41975516e+02,  8.93121694e+01,  5.90306693e+00, -3.87792660e+01
,  1.25725083e+01, -6.01675647e+00,  3.31140339e+02,  3.24499796e+02
,  2.22954821e+02,  1.12052378e+01,  4.36768719e+01,  6.52043087e+01
,  5.87132454e+00,  9.56409342e+00,  5.47609135e+00,  7.92517956e+00
,  1.27850667e+02,  3.72529845e+02,  1.74109314e+02,  1.97943411e+01
,  7.01431159e+01,  3.13297510e+02,  3.54452335e+00,  3.51806436e-01
,  2.29899425e+00, -1.99517929e+01,  2.18651058e+02,  3.78550043e+02
,  1.64779295e+02,  1.78142230e+01,  6.55614501e+01,  4.63194369e+01
,  2.16651509e+00,  2.48556222e+01,  3.48889639e+00, -1.66910072e+00
,  3.28707380e+02,  3.21715770e+02,  1.41462471e+02,  1.96490826e+01
,  6.84408808e+01,  6.21754857e+01,  4.82897213e+00,  2.94752834e+01
,  9.52075160e+00,  5.27258937e+00,  2.33579033e+02,  3.75120497e+02
,  1.82880400e+02,  1.64619048e+01,  6.44882106e+01,  3.39355279e+01
,  6.56786045e+00,  9.94404277e+00,  1.35033529e+01, -1.38274151e-01
,  1.23215596e+02,  3.46236104e+02,  1.83397199e+02,  2.14530459e+01
,  2.62522474e+02,  6.73386459e+01,  1.57359638e+01,  8.49195102e+01
, -5.08777041e+01, -1.76880453e+01,  2.00317195e+02,  3.93857854e+02
,  1.72046575e+02,  1.59015820e+01,  5.71119369e+01,  5.59415103e+01
,  1.29629130e+00,  3.13429747e+01,  3.59153732e-01, -1.11806568e+00
,  1.36498174e+02,  3.43347641e+02,  2.07544541e+02,  2.23278212e+01
,  7.49722546e+01,  1.54003760e+02,  1.23587316e+01,  6.18162259e+01
, -2.19056452e+01, -2.72982476e+01,  1.03948506e+02,  3.59762831e+02
,  1.32168004e+02,  1.79852327e+01,  7.21245692e+01,  9.42780137e+01
,  2.76488405e+00,  3.40125308e+00,  2.83495264e+00, -6.35409281e+00
,  2.27205840e+02,  4.08868307e+02,  1.43454713e+02,  1.60381763e+01
,  2.95933863e+01,  6.50908330e+01,  1.32023111e+00,  1.17952316e+01
, -1.10389181e+00,  2.45759548e-01,  1.31791615e+02,  3.19729468e+02
,  1.44179551e+02,  2.46234174e+01,  3.99862754e+02,  1.47887736e+02
,  7.56792001e+00,  1.81382134e+02, -2.98959654e+01, -1.18369287e+01
,  2.31743157e+02,  4.00905782e+02,  1.18541499e+02,  1.64727168e+01
,  7.50887779e+01,  7.43985302e+01,  2.18782101e+00,  1.09267010e+01
,  3.54912670e+00,  5.18650583e+00,  1.13700171e+02,  3.34830136e+02
,  1.18258380e+02,  1.85388195e+01,  7.91248676e+01,  5.93877380e+01
,  6.26005567e+00,  1.61187659e+01,  1.33347612e+01,  7.61875126e+00
,  1.68316790e+02,  3.77645874e+02,  1.09555526e+02,  1.74284563e+01
,  8.52775739e+01,  4.85900801e+01,  2.52547932e+00,  1.94772217e+01
,  7.50817622e+00,  3.15443841e+00]

    # start_time = timer()
    # result = scipy.optimize.minimize(model_and_image_difference.difference_with_image, guesses_list,
    #                                  method='Powell', options={'ftol':0.001,'xtol':10})
    # end_time = timer()
    # print("Iterations: " + str(result.nfev) + "    Total time: " + str(end_time - start_time) + " seconds    Time per"
    #       " iteration: " + str((end_time - start_time) / result.nfev) + " seconds")
    # if not result.success:
    #     raise ValueError("Minimization failed: " + result.message)
    #
    result_gaussians = []
    for i in range(0, len(result_x), 10):
        gaussian_params = result_x[i:i + 10]
        result_gaussians.append(Gaussian(*gaussian_params))
    return result_gaussians


def perform_gaussian_mixture_fit_from_watershed(image: ndarray, watershed_image: ndarray, out: ndarray,
                                                particles: Iterable[Particle], smooth_size: int):
    """GMM using watershed as seeds. out is a color image where the detected Gaussians can be drawn on."""

    # Out is a threshold
    out.fill(255)
    out[watershed_image == 0] = 0
    _dilate(out)

    image[out == 0] = 0
    out[...] = image
    smoothing.smooth(out, smooth_size)

    fitted = perform_gaussian_mixture_fit(out, particles_to_gaussians(out, particles))
    canvas = numpy.zeros(image.shape, dtype=numpy.float64)
    for fit in fitted:
        fit.draw(canvas)
    canvas.clip(0, 255, out=canvas)
    out[...] = canvas.astype(numpy.uint8)
    print(fitted)


def _dilate(image_3d: ndarray):
    scratch_2d = numpy.empty_like(image_3d[0])
    kernel = numpy.ones((5, 5), numpy.uint8)
    for z in range(image_3d.shape[0]):
        cv2.dilate(image_3d[z], kernel, dst=scratch_2d, iterations=2)
        image_3d[z] = scratch_2d


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
