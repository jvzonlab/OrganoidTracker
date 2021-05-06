from typing import Union, List

import numpy
import scipy.stats
from numpy import ndarray


class SlopeFinder:
    """Performs a linear regression within the given time window, to find the slope at every point the graph."""

    slope_x_values: ndarray
    slopes: ndarray
    mean_values: ndarray

    def __init__(self, x_values: Union[List[float], ndarray], y_values: Union[List[float], ndarray], *,
                 window_width: float, x_step_size: float = 1):
        if x_step_size <= 0:
            raise ValueError(f"resolution must be positive, but was {x_step_size}")
        if window_width <= 0:
            raise ValueError(f"window_width must be positive, but was {window_width}")

        extend = window_width / 2

        # Convert to numpy arrays if some other kind of array/list was given
        if not isinstance(x_values, ndarray):
            x_values = numpy.array(x_values)
        if not isinstance(y_values, ndarray):
            y_values = numpy.array(y_values)

        x_min = x_values.min()
        x_max = x_values.max()

        # Calculate the moving average for every list
        x_moving_average = list()
        y_moving_average = list()

        x = x_min
        while x <= x_max:
            # Construct a boolean area on which x values to use
            used_x_values = x_values[(x_values >= x - extend) & (x_values <= x + extend)]
            used_y_values = y_values[(x_values >= x - extend) & (x_values <= x + extend)]

            if len(used_y_values) >= 2:
                x_moving_average.append(x)
                y_moving_average.append(used_y_values.mean())

            x += x_step_size

        # Calculate the slopes
        slope_x_values = list()
        slopes = list()
        mean_values = list()
        for i in range(len(x_moving_average) - 1):
            x1 = x_moving_average[i]
            x2 = x_moving_average[i + 1]
            y1 = y_moving_average[i]
            y2 = y_moving_average[i + 1]
            slope_x_values.append((x1 + x2) / 2)
            slopes.append((y2 - y1) / (x2 - x1))
            mean_values.append((y1 + y2) / 2)

        self.slopes = numpy.array(slopes, dtype=numpy.float32)
        self.slope_x_values = numpy.array(slope_x_values, dtype=numpy.float32)
        self.mean_values = numpy.array(mean_values, dtype=numpy.float32)

