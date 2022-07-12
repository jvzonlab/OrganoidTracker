"""Some builtin image filters, so that they can be saved and loaded."""
import math
from typing import NamedTuple, Dict, Tuple, Optional

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_filters import ImageFilter


class ThresholdFilter(ImageFilter):
    """Sets all pixels below a relative threshold to zero."""

    noise_limit: float  # Scaled 0 to 1

    def __init__(self, noise_limit: float = 0.08):
        """noise_limit = 0.5 would remove all pixels of less than 50% of the max intensity of the image."""
        self.noise_limit = noise_limit

    def filter(self, time_point, image_z, image: ndarray):
        image[image < self.noise_limit * image.max()] = 0

    def copy(self) -> ImageFilter:
        return ThresholdFilter(self.noise_limit)

    def get_name(self) -> str:
        return "Threshold"


class GaussianBlurFilter(ImageFilter):
    """Applies a Gaussian blur in 2D."""

    blur_radius: int

    def __init__(self, blur_radius: int = 5):
        self.blur_radius = blur_radius

    def filter(self, time_point, image_z, image: ndarray):
        import cv2
        if len(image.shape) == 3:
            out = numpy.empty_like(image[0])
            for z in range(image.shape[0]):
                slice = image[z]
                cv2.GaussianBlur(slice, (self.blur_radius, self.blur_radius), 0, out)
                image[z] = out
        elif len(image.shape) == 2: # len(...) == 2
            out = numpy.empty_like(image)
            cv2.GaussianBlur(image, (self.blur_radius, self.blur_radius), 0, out)
            image[...] = out
        else:
            raise ValueError("Can only handle 2D or 3D images. Got shape " + str(image.shape))

    def copy(self) -> ImageFilter:
        return GaussianBlurFilter(self.blur_radius)

    def get_name(self) -> str:
        return "Gaussian blur"


class MultiplyPixelsFilter(ImageFilter):
    """Increases the brightness of all pixels."""
    factor: float

    def __init__(self, factor: float):
        if factor < 0:
            raise ValueError("factor may not be negative, but was " + str(factor))
        self.factor = factor

    def filter(self, time_point, image_z, image: ndarray):
        max_value = image.max()

        if int(self.factor) == self.factor:
            # Easy, apply cheap integer multiplication - costs almost no RAM

            # First get rid of things that will overflow
            new_max = int(max_value / self.factor)
            image[image > new_max] = new_max

            # Then do integer multiplication
            image *= int(self.factor)
            return

        # Copying required
        scaled = image * self.factor
        scaled[scaled > max_value] = max_value  # Prevent overflow
        image[...] = scaled.astype(numpy.uint8)

    def copy(self):
        return MultiplyPixelsFilter(self.factor)

    def get_name(self) -> str:
        return "Increase brightness"


class IntensityPoint(NamedTuple):
    time_point: TimePoint
    z: int

    def distance_squared(self, other: "IntensityPoint") -> int:
        return (self.time_point.time_point_number() - other.time_point.time_point_number()) ** 2 \
               + (self.z - other.z) ** 2


class InterpolatedMinMaxFilter(ImageFilter):
    """Allows you to set the min/max pixel values at different points during the time-lapse. For all other points,
    the min and max values are interpolated."""


    points: Dict[IntensityPoint, Tuple[float, float]]  # Dictionary of point to (min, max)

    def __init__(self, points_to_min_max: Optional[Dict[IntensityPoint, Tuple[float, float]]] = None):
        if points_to_min_max is None:
            points_to_min_max = dict()
        self.points = points_to_min_max

    def interpolate_point(self, search_point: IntensityPoint) -> Optional[Tuple[float, float]]:
        if len(self.points) == 0:
            return None  # Cannot return anything
        if len(self.points) == 1:
            return list(self.points.values())[0]  # Return the only point

        # Find the closest two points
        closest_point = None
        second_closest_point = None
        closest_distance_squared = 1_000_000_000
        second_closest_distance_squared = 1_000_000_000

        for point in self.points.keys():
            distance_squared = point.distance_squared(search_point)
            if distance_squared < closest_distance_squared:
                second_closest_point = closest_point
                second_closest_distance_squared = closest_distance_squared
                closest_point = point
                closest_distance_squared = distance_squared
            elif distance_squared < second_closest_distance_squared:
                second_closest_distance_squared = distance_squared
                second_closest_point = point

        closest_distance = math.sqrt(closest_distance_squared)
        second_closest_distance = math.sqrt(second_closest_distance_squared)
        sum_distance = closest_distance + second_closest_distance

        # Do linear interpolation of min and max intensity
        closest_min_intensity, closest_max_intensity = self.points[closest_point]
        second_closest_min_intensity, second_closest_max_intensity = self.points[second_closest_point]

        min_intensity = closest_min_intensity * second_closest_distance / sum_distance + second_closest_min_intensity * closest_distance / sum_distance
        max_intensity = closest_max_intensity * second_closest_distance / sum_distance + second_closest_max_intensity * closest_distance / sum_distance
        return min_intensity, max_intensity

    def filter(self, time_point: TimePoint, image_z: Optional[int], image: ndarray):
        if len(image.shape) == 3:
            # 3D image, work layer by layer
            for z in range(image.shape[0]):
                self._filter_2d(time_point, z, image[z])
            return
        self._filter_2d(time_point, image_z, image)

    def _filter_2d(self, time_point: TimePoint, image_z: int, image: ndarray):
        interpolation_result = self.interpolate_point(IntensityPoint(time_point=time_point, z=int(image_z)))
        if interpolation_result is None:
            return
        min_value, max_value = interpolation_result

        # Need to cast to int if we're working with int arrays
        if numpy.issubdtype(image.dtype, numpy.integer):
            min_value = int(min_value)
            max_value = int(max_value)

        # Change the min
        image[image < min_value] = min_value
        image -= min_value
        max_value -= min_value

        # Change the max
        image[image > max_value] = max_value

    def copy(self) -> "InterpolatedMinMaxFilter":
        return InterpolatedMinMaxFilter(self.points.copy())

    def get_name(self) -> str:
        return "Interpolated min/max"

    def get_point_exact(self, point: IntensityPoint) -> Optional[Tuple[float, float]]:
        """Gets the min/max value that was stored for the given point. If there is no value stored for that point, then
        this method returns None. (So it won't interpolate between points, unlike self.interpolate_point()."""
        return self.points.get(point)

    def set_point(self, point: IntensityPoint, value: Optional[Tuple[float, float]]):
        """Sets a point, or deletes it if the value is None."""
        if value is None:
            if point in self.points:
                del self.points[point]
        else:
            self.points[point] = float(value[0]), float(value[1])
