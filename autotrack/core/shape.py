import cv2
import math
from typing import Optional, List, Tuple

from matplotlib.axes import Axes
from matplotlib.patches import Ellipse as mpl_Ellipse
from numpy import ndarray

from autotrack.core.ellipse import EllipseStack, Ellipse
from autotrack.core.gaussian import Gaussian


class ParticleShape:
    """Represents the shape of a particle. No absolute position data is stored here."""

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        """Draws a shape in 2d, at the given x and y. dz is how many z layers we are removed from the actual position,
        dt the same, but for time steps. Both dz and dt can be negative. area is the drawing surface, and color is the
        desired color.
        """
        raise NotImplementedError()

    def draw3d_color(self, x: float, y: float, z: float, dt: int, image: ndarray, color: Tuple[float, float, float]):
        """Draws a shape in 3d to a color image."""
        raise NotImplementedError()

    @staticmethod
    def default_draw3d_color(x: float, y: float, z: float, dt: int, image: ndarray,
                             color: Tuple[float, float, float], radius_xy=5, radius_z=0):
        min_x, min_y, min_z = int(x - radius_xy), int(y - radius_xy), int(z - radius_z)
        max_x, max_y, max_z = int(x + radius_xy + 1), int(y + radius_xy + 1), int(z + radius_z + 1)
        image[min_z:max_z, min_y:max_y, min_x:max_x, 0] = color[0]
        image[min_z:max_z, min_y:max_y, min_x:max_x, 1] = color[1]
        image[min_z:max_z, min_y:max_y, min_x:max_x, 2] = color[2]

    def is_unknown(self) -> bool:
        """Returns True if there is no shape information available at all."""
        return False

    def to_list(self) -> List:
        """Converts this shape to a list for serialization purposes. Convert back using from_list"""
        raise NotImplementedError()

    def __repr__(self):
        return "<" + type(self).__name__ + ">"

    def volume(self) -> float:
        """Gets a 3-dimensional volume, in pixels^3"""
        raise NotImplementedError()

    def intensity(self) -> float:
        """Gets the maximum intensity of the shape, on a scale of 0 to 1"""
        raise NotImplementedError()


class EllipseShape(ParticleShape):
    """Represents an ellipsoidal shape. The shape only stores 2D information. This class was previously used to store
    some primitive shape information. The class is still present, so that you are able to load old data."""
    _ellipse: Ellipse

    _original_perimeter: float
    _original_area: float
    _eccentric: bool

    def __init__(self, ellipse_dx: float, ellipse_dy: float, ellipse_width: float, ellipse_height: float,
                 ellipse_angle: float, original_perimeter: float, original_area: float, eccentric: bool = False):
        self._ellipse = Ellipse(ellipse_dx, ellipse_dy, ellipse_width, ellipse_height, ellipse_angle)

        self._original_perimeter = original_perimeter
        self._original_area = original_area
        self._eccentric = eccentric

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        if abs(dz) > 3:
            return
        fill = dt == 0
        edgecolor = 'white' if fill else color
        alpha = max(0.1, 0.5 - abs(dz / 6))
        area.add_artist(mpl_Ellipse(xy=(x + self._ellipse.x, y + self._ellipse.y),
                                    width=self._ellipse.width, height=self._ellipse.height, angle=self._ellipse.angle,
                                    fill=fill, facecolor=color, edgecolor=edgecolor, linestyle="dashed", linewidth=1,
                                    alpha=alpha))

    def draw3d_color(self, x: float, y: float, z: float, dt: int, image: ndarray, color: Tuple[float, float, float]):
        if dt != 0:
            return
        min_z = max(0, int(z) - 4)
        max_z = min(image.shape[0], int(z) + 4 + 1)
        thickness = -1 if dt == 0 else 1  # thickness == -1 causes ellipse to be filled
        for z_layer in range(min_z, max_z):
            dz = abs(int(z) - z_layer) + 1
            z_color = (color[0] / dz, color[1] / dz, color[2] / dz)
            self._draw_to_image(image[z_layer], x, y, z_color, thickness)

    def _draw_to_image(self, image_2d: ndarray, x: float, y: float, color: Tuple[float, float, float], thickness: int):
        # PyCharm cannot recognize signature of cv2.ellipse, so the warning is a false positive:
        # noinspection PyArgumentList
        cv2.ellipse(image_2d, ((x + self._ellipse.x, y + self._ellipse.y),
                                   (self._ellipse.width, self._ellipse.height), self._ellipse.angle),
                    color=color, thickness=thickness)

    def volume(self) -> float:
        return self._original_area

    def to_list(self) -> List:
        return ["ellipse", self._ellipse.x, self._ellipse.y, self._ellipse.width, self._ellipse.height,
                self._ellipse.angle, self._original_perimeter, self._original_area, bool(self._eccentric)]


class UnknownShape(ParticleShape):

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        draw_marker_2d(x, y, dz, dt, area, color)

    def draw3d_color(self, x: float, y: float, z: float, dt: int, image: ndarray, color: Tuple[float, float, float]):
        self.default_draw3d_color(x, y, z, dt, image, color)

    def volume(self) -> float:
        return 0

    def is_unknown(self) -> bool:
        return True

    def to_list(self):
        return []


class GaussianShape(ParticleShape):
    """Represents a particle represented by a Gaussian shape."""
    _gaussian: Gaussian

    def __init__(self, gaussian: Gaussian):
        self._gaussian = gaussian

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        draw_marker_2d(x, y, dz, dt, area, color)

    def draw3d_color(self, x: float, y: float, z: float, dt: int, image: ndarray, color: Tuple[float, float, float]):
        self._gaussian.translated(x, y, z).draw_colored(image, color)

    def to_list(self) -> List:
        return ["gaussian", *self._gaussian.to_list()]

    def volume(self) -> float:
        """We take the volume of a 3D ellipse with radii equal to the Gaussian variances."""
        return 4/3 * math.pi * \
               math.sqrt(self._gaussian.cov_xx) * math.sqrt(self._gaussian.cov_yy) * math.sqrt(self._gaussian.cov_zz)

    def intensity(self) -> float:
        return self._gaussian.a / 256


class EllipseStackShape(ParticleShape):
    """A stack of ellipses. Fairly simple, but still a useful 3D representation of the particle shape."""

    _ellipse_stack: EllipseStack
    _center_ellipse: int  # Z position of the particle center

    def __init__(self, ellipse_stack: EllipseStack, center_ellipse: int):
        self._ellipse_stack = ellipse_stack
        self._center_ellipse = center_ellipse

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: str):
        draw_marker_2d(x, y, dz, dt, area, color)
        fill = dt == 0
        edgecolor = 'white' if fill else color
        ellipse = self._ellipse_stack.get_ellipse(self._center_ellipse + dz)
        if ellipse is None:
            return

        area.add_artist(mpl_Ellipse(xy=(x + ellipse.x, y + ellipse.y),
                                    width=ellipse.width, height=ellipse.height, angle=ellipse.angle, alpha=0.5,
                                    fill=fill, facecolor=color, edgecolor=edgecolor, linestyle="dashed", linewidth=1))

    def draw3d_color(self, x: float, y: float, z: float, dt: int, image: ndarray, color: Tuple[float, float, float]):
        lowest_ellipse_z = int(z) - self._center_ellipse
        for layer_z in range(len(image)):
            ellipse = self._ellipse_stack.get_ellipse(layer_z - lowest_ellipse_z)
            if ellipse is not None:
                ellipse.draw_to_image(image[layer_z], color, dx=x, dy=y, filled=True)

    def to_list(self) -> List:
        list = ["ellipse_stack", self._center_ellipse]
        for ellipse in self._ellipse_stack:
            if ellipse is None:
                list.append(None)
            else:
                list.append([ellipse.x, ellipse.y, ellipse.width, ellipse.height, ellipse.angle])
        return list

    def volume(self) -> float:
        """Volume is in pixels, so we can just return the summed area of the ellipses."""
        volume = 0
        for ellipse in self._ellipse_stack:
            volume += ellipse.area()
        return volume


def draw_marker_2d(x: float, y: float, dz: int, dt: int, area: Axes, color: str):
    """The default (point) representation of a shape. Implementation can fall back on this if they want."""
    if abs(dz) > 3:
        return
    marker_style = 's' if dz == 0 else 'o'
    marker_size = max(1, 7 - abs(dz) - abs(dt))
    area.plot(x, y, marker_style, color=color, markeredgecolor='black', markersize=marker_size, markeredgewidth=1)


def from_list(list: List) -> ParticleShape:
    if len(list) == 0:
        return UnknownShape()
    type = list[0]
    if type == "ellipse":
        return EllipseShape(*list[1:])
    elif type == "gaussian":
        return GaussianShape(Gaussian(*list[1:]))
    elif type == "ellipse_stack":
        ellipses = []
        for ellipse in list[2:]:
            if ellipse is None:
                ellipses.append(None)
            else:
                ellipses.append(Ellipse(*ellipse))
        return EllipseStackShape(EllipseStack(ellipses), list[1])
    raise ValueError("Cannot deserialize " + str(list))