from typing import Tuple, Union

import numpy
from numpy import ndarray
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Ellipsoid:
    _center: ndarray
    _radii_squared: ndarray
    _radii: ndarray
    _rotation_T: ndarray  # Transposed rotation matrix

    def __init__(self, center: ndarray, radii: ndarray, rotation: ndarray):
        self._center = center.astype(dtype=numpy.float32)
        if numpy.any(numpy.isnan(self._center)):
            raise ValueError("NaN in center: " + str(center))
        self._radii = radii.astype(dtype=numpy.float32)
        if numpy.any(numpy.isnan(self._radii)):
            raise ValueError("NaN in radii: " + str(radii))
        if numpy.any(self._radii > 100):
            raise ValueError(">100 in radii: " + str(radii))
        self._radii_squared = self._radii ** 2
        self._rotation_T = rotation.astype(dtype=numpy.float32).transpose()

    def draw_to_image(self, image: ndarray, fill_color: int):
        max_radius = self._radii.max()
        offset_x = max(0, int(self._center[0] - max_radius))
        offset_y = max(0, int(self._center[1] - max_radius))
        offset_z = max(0, int(self._center[2] - max_radius))

        sub_image = image[offset_z:int(self._center[2] + max_radius),
                    offset_y:int(self._center[1] + max_radius),
                    offset_x:int(self._center[0] + max_radius)
                    ]
        for index in numpy.ndindex(sub_image.shape):
            x = index[2] + offset_x
            y = index[1] + offset_y
            z = index[0] + offset_z

            if self.is_at_border((x, y, z)):
                sub_image[index] = fill_color

    def is_at_border(self, x_y_z: Union[ndarray, Tuple[float, float, float]]):
        translated_x_y_z = x_y_z - self._center
        rotated_x_y_z = self._rotation_T @ translated_x_y_z
        rotated_x_y_z_squared = rotated_x_y_z**2
        division = rotated_x_y_z_squared / self._radii_squared
        return abs(1 - numpy.sum(division)) < 0.1

def perform_fit(watershedded_image: ndarray, buffer_image: ndarray):
    buffer_image.fill(0)

    area_count = watershedded_image.max()
    for i in range(area_count):
        z_coords, y_coords, x_coords = numpy.where(watershedded_image == i)
        if len(x_coords) == 0:
            continue
        center, radii, evecs, v = _ellipsoid_fit(x_coords, y_coords, z_coords)
        print("CENTER: " + str(center))
        print("RADII: " + str(radii))
        print("EVECS: " + str(evecs))
        try:
            Ellipsoid(center, radii, evecs).draw_to_image(buffer_image, 255)
        except ValueError as e:
            print(e)



def ellipsoid_plot(center, radii, rotation, ax, plotAxes=False, cageColor='b', cageAlpha=0.2):
    """Plot an ellipsoid"""

    u = numpy.linspace(0.0, 2.0 * numpy.pi, 100)
    v = numpy.linspace(0.0, numpy.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * numpy.outer(numpy.cos(u), numpy.sin(v))
    y = radii[1] * numpy.outer(numpy.sin(u), numpy.sin(v))
    z = radii[2] * numpy.outer(numpy.ones_like(u), numpy.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = numpy.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = numpy.array([[radii[0], 0.0, 0.0],
                         [0.0, radii[1], 0.0],
                         [0.0, 0.0, radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = numpy.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = numpy.linspace(-p[0], p[0], 100) + center[0]
            Y3 = numpy.linspace(-p[1], p[1], 100) + center[1]
            Z3 = numpy.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)


# Based on the MIT-licensed code by Aleksandr Bazhin
# https://github.com/aleksandrbazhin/ellipsoid_fit_python/blob/3a29f8d9321f03e5580fd85d42e0c55f68ead97d/ellipsoid_fit.py#L122
# which was based on http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
def _ellipsoid_fit(x: ndarray, y: ndarray, z: ndarray):
    """Fits an ellipsoid to the list of points [[x,y,z], [x,y,z], [x,y,z],...]"""
    D = numpy.array([x * x,
                     y * y,
                     z * z,
                     2 * x * y,
                     2 * x * z,
                     2 * y * z,
                     2 * x,
                     2 * y,
                     2 * z])
    DT = D.conj().T
    v = numpy.linalg.solve(D.dot(DT), D.dot(numpy.ones(numpy.size(x))))
    A = numpy.array([[v[0], v[3], v[4], v[6]],
                     [v[3], v[1], v[5], v[7]],
                     [v[4], v[5], v[2], v[8]],
                     [v[6], v[7], v[8], -1]])

    center = numpy.linalg.solve(- A[:3, :3], [[v[6]], [v[7]], [v[8]]])
    T = numpy.eye(4)
    T[3, :3] = center.T
    R = T.dot(A).dot(T.conj().T)
    evals, evecs = numpy.linalg.eig(R[:3, :3] / -R[3, 3])
    radii = numpy.sqrt(1. / evals)
    return center.ravel(), radii, evecs, v
