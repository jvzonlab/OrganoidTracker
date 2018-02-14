from numpy import ndarray
import numpy


class Image2d:
    """Wrapper around an ndarray that makes working with images easier."""
    _image_data: ndarray # Indexed as [y][x]
    _max_intensity: float # Maximum intensity (like 4095), used to scale the output intensities between 0 and 1

    def __init__(self, image_data: ndarray):
        """Creates an instance from the given Numpy array"""
        self._image_data = image_data
        self._max_intensity = numpy.amax(image_data)

    def get_average_intensity_at(self, center_x: int, center_y: int, r: int = 1):
        total_intensity = 0
        for x in range(center_x - r, center_x + r):
            for y in range(center_y - r, center_y + r):
                total_intensity += self._image_data[y][x] / self._max_intensity

        return total_intensity / ((r * 2 + 1) ** 2)


    def get_intensities_square(self, center_x: int, center_y: int, r: int, out_array: ndarray = None) -> ndarray:
        """Imagine a square centered around (center_x, center_y) with radius r. This method returns all intensities at
        the edge of this square. Four arrays are returned, representing the intensities at the top, right, bottom and
        left side of the square.

        out_array must be of size `s = (4, 2 * r)`. If no array is given, it is automatically created.
        """

        if out_array is None:
            out_array = numpy.empty((4, 2 * r))

        # Top and bottom rows
        y_top = center_y - r
        y_bottom = center_y + r
        for i in range(2 * r):
            x = center_x - r + i
            out_array[0][i] = self._image_data[y_top][x] / self._max_intensity
            out_array[2][i] = self._image_data[y_bottom][x] / self._max_intensity

        # Left and right rows
        x_left = center_x - r
        x_right = center_x + r
        for i in range(2 * r):
            y = center_y - r + i
            out_array[1][i] = self._image_data[y][x_left] / self._max_intensity
            out_array[3][i] = self._image_data[y][x_right] / self._max_intensity

        return out_array
