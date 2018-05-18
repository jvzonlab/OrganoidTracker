import scipy.optimize
import numpy, numpy.linalg
from numpy import ndarray
import mahotas.labeled
from typing import List, Tuple, Iterable, Union, Optional


class Gaussian:
    """A Gaussian function (also called a normal distribution) in 3D."""
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

    def __repr__(self):
        vars = dict(self.__dict__)
        del vars["_draw_xrange"]
        del vars["_draw_yrange"]
        del vars["_draw_zrange"]
        return "Gaussian(**" + repr(vars) + ")"


def _gaussian_3d_single(pos: ndarray, a: float, mu: ndarray, cov_inv: ndarray):
    """Returns the value of the given Gaussian at the specified position.
    pos: xyz position to get the value at, a: maximum value of the Gaussian, mu: xyz mean position of the Gaussian,
    cov_inv: covariance matrix."""
    return a * numpy.exp(-1 / 2 * ((pos - mu) @ cov_inv @ (pos - mu)))


class Fitting:
    """A class used to fit multiple Gaussians at once to an image. Everything happens in 3D.

    The class stores a list of Gaussians. Then, during a fit, several of the parameters of the Gaussian are changed,
    such that the Gaussians better resemble the reference image.

    The fit requires input values and output values, which are the pixel positions and values, respectively. The
    parameters are the Gaussian parameters, i.e. their mean, covariance matrix and prefactor. Scipy wants all of these
    values in 1D, so we have to massage our data a bit to get our 3D images in that format. Numpy's ravel() function is
    really useful for that.
    """
    _gaussians: List[Gaussian]
    _reference_image: ndarray  # Original measurement data
    _model_image: ndarray  # Output of drawing all Gaussians on an empty image
    _model_needs_redraw: bool = True  # True when the model_image is not updated for changes to the Gaussians

    def __init__(self, gaussians: List[Gaussian], reference_image: ndarray, scratch_image: ndarray):
        self._gaussians = gaussians
        self._reference_image = reference_image
        self._model_image = scratch_image
        if reference_image.shape != scratch_image.shape:
            raise ValueError("Shapes of reference and scratch image are not equal")

    # INTERNAL MEAN FITTING FUNCTIONS

    def _update_means(self, xyz_positions: Union[Tuple[float, ...], ndarray]):
        """Updates the positions of the Gaussians in this class using a list of x,y,z values."""
        for i in range(0, len(xyz_positions), 3):
            gaussian = self._gaussians[int(i / 3)]
            gaussian.mu_x = xyz_positions[i]
            gaussian.mu_y = xyz_positions[i + 1]
            gaussian.mu_z = xyz_positions[i + 2]
        self._model_needs_redraw = True

    def _fit_means_func(self, image_size: Tuple[int, int, int], *gaussian_positions):
        """Called by scipy to perform the fitting."""
        self._update_means(gaussian_positions)

        image = numpy.empty(image_size, dtype=numpy.uint8)
        for gaussian in self._gaussians:
            gaussian.add_to_image(image)
        return image.ravel()

    def _get_means(self) -> List[float]:
        positions = []
        for gaussian in self._gaussians:
            positions.append(gaussian.mu_x)
            positions.append(gaussian.mu_y)
            positions.append(gaussian.mu_z)
        return positions

    # INTERNAL COVARIANCE FITTING FUNCTIONS

    def _fit_covariances_func(self, gaussian: Gaussian, cov_xx, cov_yy, cov_zz, cov_xy, cov_xz, cov_yz):
        """Called by scipy to perform the fitting."""
        gaussian.cov_xx = cov_xx
        gaussian.cov_yy = cov_yy
        gaussian.cov_zz = cov_zz
        gaussian.cov_xy = cov_xy
        gaussian.cov_xz = cov_xz
        gaussian.cov_yz = cov_yz

        image = self._model_image.astype(dtype=numpy.float64)
        gaussian.add_to_image(image)
        value = image.ravel()
        return value

    # OTHER PLUMBING

    def _draw(self, excluded: Optional[Gaussian] = None):
        self._model_image.fill(0)
        for gaussian in self._gaussians:
            if gaussian != excluded:
                gaussian.add_to_image(self._model_image)
        self._model_needs_redraw = False

    def _get_positions(self) -> Tuple[ndarray, ndarray, ndarray]:
        """Gets the x,y,z positions in iteration order."""
        x = numpy.arange(0, self._reference_image.shape[2])
        y = numpy.arange(0, self._reference_image.shape[1])
        z = numpy.arange(0, self._reference_image.shape[0])
        x, y, z = numpy.meshgrid(x, y, z)
        return x, y, z

    def get_loss(self) -> int:
        """Gets the so-called loss, which is the squared difference between the reference data and the fitted data."""
        if self._model_needs_redraw:
            self._draw()
        loss = 0
        for position in numpy.ndindex(self._reference_image):
            loss += (self._reference_image[position] - self._model_image[position]) ** 2
        return loss

    def get_image(self) -> ndarray:
        """Gets access to the image as drawn by the Gaussians. Note that this method does not return a copy, so the
        image can be changed later if you continue fitting."""
        if self._model_needs_redraw:
            self._draw()
        return self._model_image

    # PUBLIC FITTING FUNCTIONS

    def fit_means(self):
        # noinspection PyTypeChecker
        optimized_means, uncertainty_covariance = scipy.optimize.curve_fit(
            self._fit_means_func, self._reference_image.shape, self._reference_image.ravel(), p0=self._get_means())
        self._update_means(optimized_means)

    def fit_covariance(self):
        print("Fitting covariance...")
        for i in range(len(self._gaussians)):
            gaussian = self._gaussians[i]
            print("Picked Gaussian: " + str(gaussian))
            self._draw(excluded=gaussian)
            positions = self._reference_image.shape
            reference = self._reference_image.ravel()
            covariances = [gaussian.cov_xx, gaussian.cov_yy, gaussian.cov_zz, gaussian.cov_xy, gaussian.cov_xz, gaussian.cov_yz]
            print("Positions:" + str(positions) + " Reference image: " + str(len(reference)) + " Parameters: " + str(len(covariances)))
            # noinspection PyTypeChecker
            optimized_covariances, uncertainty_covariance = scipy.optimize.curve_fit(
                self._fit_covariances_func, gaussian, reference, p0=covariances)

            gaussian.cov_xx = optimized_covariances[0]
            gaussian.cov_yy = optimized_covariances[1]
            gaussian.cov_zz = optimized_covariances[2]
            gaussian.cov_xy = optimized_covariances[3]
            gaussian.cov_xz = optimized_covariances[4]
            gaussian.cov_yz = optimized_covariances[5]
            self._model_needs_redraw = True
            print("Done! Changed to " + str(gaussian))
        print("All done!")



def _initial_gaussians(blurred_image: ndarray, watershedded_image: ndarray) -> List[Gaussian]:
    gaussians = []
    centers = mahotas.center_of_mass(blurred_image, watershedded_image)
    for i in range(1, centers.shape[0]):
        center = centers[i]
        if numpy.isnan(center[0]):
            continue
        x, y, z = center[2], center[1], center[0]
        max = blurred_image[int(z), int(y), int(x)]
        gaussians.append(Gaussian(max, x, y, z, cov_xx=20, cov_yy=20, cov_zz=2,
                                  cov_xy=0, cov_xz=0, cov_yz=0))
    return gaussians


def intialize_fit(blurred_image: ndarray, watershedded_image: ndarray, buffer_image: ndarray) -> Fitting:
    """Extracts some initial Gaussians from a Watershed result so that they can be fitted."""
    gaussians = _initial_gaussians(blurred_image, watershedded_image)
    return Fitting(gaussians, blurred_image, buffer_image)


#scipy.optimize.curve_fit(gaussian_3d)