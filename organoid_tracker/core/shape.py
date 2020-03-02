import cv2
import math
from typing import Optional, List, Tuple, Any

import numpy
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse as mpl_Ellipse
from numpy import ndarray

from organoid_tracker.core.ellipse import EllipseStack, Ellipse
from organoid_tracker.core.gaussian import Gaussian
from organoid_tracker.core.mask import Mask
from organoid_tracker.core.typing import MPLColor


class ParticleShape:
    """Represents the shape of a position. No absolute position data is stored here."""
    _DEFAULT_GAUSSIAN = Gaussian(300, 0, 0, 0, 15, 15, 2, 0, 0, 0)

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: MPLColor, edge_color: MPLColor):
        """Draws a shape in 2d, at the given x and y. dz is how many z layers we are removed from the actual position,
        dt the same, but for time steps. Both dz and dt can be negative. area is the drawing surface, and color is the
        desired color.
        """
        raise NotImplementedError()

    def draw_marker_2d(self, x, y, dz, dt, area, color, edge_color):
        """Draws a simple marker."""
        draw_marker_2d(x, y, dz, dt, area, color, edge_color)

    def draw3d_color(self, x: float, y: float, z: float, dt: int, image: ndarray, color: Tuple[float, float, float]):
        """Draws a shape in 3d to a color image."""
        raise NotImplementedError()

    @staticmethod
    def default_draw3d_color(x: float, y: float, z: float, dt: int, image: ndarray,
                             color: Tuple[float, float, float], radius_xy=5, radius_z=0):
        ParticleShape._DEFAULT_GAUSSIAN.translated(x, y, z).draw_colored(image, color)

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
        """Gets the maximum intensity of the shape, on a scale of 0 to 1. Raises ValueError() if unknown."""
        raise ValueError()

    def ellipse(self) -> Ellipse:
        """Gets an ellipse describing the shape. Any 3D info is lost."""
        return Ellipse(0, 0, 20, 20, 0)

    def draw_mask(self, mask: Mask, x: float, y: float, z: float):
        """Draws a mask on the given drawing area."""
        mask.set_bounds_around(x, y, z, 20, 20, 0)
        mask_array = mask.get_mask_array()
        if len(mask_array) == 1:
            cv2.circle(mask_array[0], (int(x - mask.offset_x), int(y - mask.offset_y)), color=1, radius=20, thickness=cv2.FILLED)

    def is_failed(self) -> bool:
        """Returns True if an attempt was made to detemine the shape of the position, but for whatever reason it
         failed."""
        return False


class EllipseShape(ParticleShape):
    """Represents an ellipsoidal shape. The shape only stores 2D information. This class was previously used to store
    some primitive shape information. The class is still present, so that you are able to load old data."""
    _ellipse: Ellipse

    _original_perimeter: float
    _original_area: float
    _eccentric: bool

    def __init__(self, ellipse_dx: float, ellipse_dy: float, ellipse_width: float, ellipse_height: float,
                 ellipse_angle: float, original_perimeter: float = -1, original_area: float = -1, eccentric: bool = False):
        self._ellipse = Ellipse(ellipse_dx, ellipse_dy, ellipse_width, ellipse_height, ellipse_angle)

        if original_perimeter == -1:
            original_perimeter = self._ellipse.perimeter()
        if original_area == -1:
            original_perimeter = self._ellipse.area()
        self._original_perimeter = original_perimeter
        self._original_area = original_area
        self._eccentric = eccentric

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: MPLColor, edge_color: MPLColor):
        if abs(dz) > 3:
            return
        fill = dt == 0
        edgecolor = edge_color if fill else color
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

    def _draw_to_image(self, image_2d: ndarray, x: float, y: float, color: Any, thickness: int):
        # PyCharm cannot recognize signature of cv2.ellipse, so the warning is a false positive:
        # noinspection PyArgumentList
        cv2.ellipse(image_2d, ((x + self._ellipse.x, y + self._ellipse.y),
                                   (self._ellipse.width, self._ellipse.height), self._ellipse.angle),
                    color=color, thickness=thickness)

    def draw_mask(self, mask: Mask, x: float, y: float, z: float):
        min_x, min_y, max_x, max_y = self._ellipse.get_rectangular_bounds()
        mask.set_bounds_exact(x + min_x, y + min_y, z, x + max_x, y + max_y, z + 1)
        mask_array = mask.get_mask_array()
        self._ellipse.draw_to_image(mask_array[int(z - mask.offset_z)], 1, dx=x - mask.offset_x, dy=y - mask.offset_y, filled=True)

    def volume(self) -> float:
        return self._original_area

    def ellipse(self) -> Ellipse:
        return self._ellipse

    def to_list(self) -> List:
        return ["ellipse", self._ellipse.x, self._ellipse.y, self._ellipse.width, self._ellipse.height,
                self._ellipse.angle, self._original_perimeter, self._original_area, bool(self._eccentric)]


class UnknownShape(ParticleShape):
    """Used when no shape was found. Don't use `isinstance(obj, UnknownShape)`, just use `obj.is_unknown()`.
    It is not necessary to create new instances of this object, just use the UNKNOWN_SHAPE constant."""

    _is_failed: bool

    def __init__(self, *, is_failed: bool):
        self._is_failed = is_failed

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: MPLColor, edge_color: MPLColor):
        area.plot(x, y, 'o', markersize=25, color=(0, 0, 0, 0), markeredgecolor=color, markeredgewidth=5)

    def draw3d_color(self, x: float, y: float, z: float, dt: int, image: ndarray, color: Tuple[float, float, float]):
        self.default_draw3d_color(x, y, z, dt, image, color)

    def volume(self) -> float:
        return 0

    def is_unknown(self) -> bool:
        return True

    def to_list(self):
        if self._is_failed:
            return ["failed"]
        return []

    def is_failed(self) -> bool:
        return self._is_failed


# No need to create new unknown shape - any instance is the same after all. Just use this instance.
UNKNOWN_SHAPE = UnknownShape(is_failed=False)
FAILED_SHAPE = UnknownShape(is_failed=True)


class GaussianShape(ParticleShape):
    """Represents a position represented by a Gaussian shape."""
    _gaussian: Gaussian

    def __init__(self, gaussian: Gaussian):
        self._gaussian = gaussian

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: MPLColor, edge_color: MPLColor):
        dz_for_gaussian = int(dz - self._gaussian.mu_z)
        ellipse = self.ellipse()
        fill = False
        alpha = max(0.2, 0.8 - abs(dz_for_gaussian / 6))
        area.add_artist(mpl_Ellipse(xy=(x + ellipse.x, y + ellipse.y),
                                    width=ellipse.width, height=ellipse.height, angle=ellipse.angle,
                                    fill=fill, edgecolor=color, linestyle="dashed", linewidth=2,
                                    alpha=alpha))


    def draw3d_color(self, x: float, y: float, z: float, dt: int, image: ndarray, color: Tuple[float, float, float]):
        self._gaussian.translated(x, y, z).draw_colored(image, color)

    def to_list(self) -> List:
        return ["gaussian", *self._gaussian.to_list()]

    def volume(self) -> float:
        """We take the volume of a 3D ellipse with radii equal to the Gaussian variances."""
        return 4/3 * math.pi * \
               math.sqrt(self._gaussian.cov_xx) * math.sqrt(self._gaussian.cov_yy) * math.sqrt(self._gaussian.cov_zz)

    def draw_mask(self, mask: Mask, x: float, y: float, z: float):
        drawing_gaussian = self._gaussian.translated(x, y, z)
        mask.set_bounds(drawing_gaussian.get_bounds())
        drawing_gaussian.translated(-mask.offset_x, -mask.offset_y, -mask.offset_z).draw_ellipsoid(mask.get_mask_array())

    def intensity(self) -> float:
        return self._gaussian.a / 256

    def ellipse(self) -> Ellipse:
        ellipse = self._gaussian.to_ellipse()
        if max(ellipse.width, ellipse.height) / min(ellipse.width, ellipse.height) > 1000 or ellipse.height > 10000\
                or ellipse.width > 10000:
            return super().ellipse()  # Cannot draw this ellipse
        return ellipse

    def __repr__(self):
        return "GaussianShape(" + repr(self._gaussian) + ")"


class EllipseStackShape(ParticleShape):
    """A stack of ellipses. Fairly simple, but still a useful 3D representation of the position shape."""

    _ellipse_stack: EllipseStack
    _center_ellipse: int  # Z position of the position center

    def __init__(self, ellipse_stack: EllipseStack, center_ellipse: int):
        self._ellipse_stack = ellipse_stack
        self._center_ellipse = center_ellipse

    def draw2d(self, x: float, y: float, dz: int, dt: int, area: Axes, color: MPLColor, edge_color: MPLColor):
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

    def ellipse(self) -> Ellipse:
        return self._ellipse_stack[self._center_ellipse]


def draw_marker_2d(x: float, y: float, dz: int, dt: int, area: Axes, color: MPLColor, edge_color: MPLColor):
    """The default (point) representation of a shape. Implementation can fall back on this if they want."""
    if abs(dz) > 3:
        return
    marker_style = 's' if dz == 0 else 'o'
    marker_size = max(1, 7 - abs(dz) - abs(dt))
    area.plot(x, y, marker_style, color=color, markeredgecolor=edge_color, markersize=marker_size, markeredgewidth=1)


def from_list(list: List) -> ParticleShape:
    if len(list) == 0:
        return UNKNOWN_SHAPE
    type = list[0]
    if type == "failed":
        return FAILED_SHAPE
    elif type == "ellipse":
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
