from abc import abstractmethod, ABC
from typing import List, Union, Tuple

import numpy
from matplotlib.axes import Axes
from numpy import ndarray

from organoid_tracker.core import min_none, max_none


class PlotAverage(ABC):

    @abstractmethod
    def plot(self, axes: Axes, *, color="blue", linewidth=2, error_opacity=0.8, standard_error: bool = False, label="Moving average"):
        """Plots the moving average to the given axis. For example:

                    plot(plt.gca(), label="My moving average")

                The error bar is either the standard deviation (if standard_error is False) or the standard error (if standard_error is True).
                """


class MovingAverage(PlotAverage):
    """A moving average calculation for a point cloud. Usage example:

    >>> import matplotlib.pyplot as plt
    >>> # Create and plot some sample data
    >>> x_values = [0, 0, 1, 2, 4, 6, 8]
    >>> y_values = [6, 8, 5, 3, 2, -1, -4]
    >>> plt.scatter(x_values, y_values)
    >>>
    >>> # Calculate and plot the moving average
    >>> average = MovingAverage(x_values, y_values, window_width=3)  # Calculate with this window width
    >>> average.plot(plt.gca())  # Plot to the current axis (gca = get current axis)
    >>> plt.show()
    """

    x_values: ndarray  # X values of the mean
    mean_values: ndarray  # Y values of the mean. You can plot the mean like plt.plot(x_values, mean_values)
    standard_deviation_values: ndarray  # Standard deviation in the mean. Useful for plt.fill_between.
    counts_in_standard_deviation_values: ndarray  # Counts used for calculating the standard deviation

    def __init__(self, x_values: Union[List[float], ndarray], y_values: Union[List[float], ndarray], *,
                 window_width: float, x_step_size: float = 1):
        """Calculates the moving average with the given width. The resolution determines the step size in which the
        average is moved over the data. The lower the value, the longer the calculation takes."""
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

        # Setup output lists
        x_moving_average = list()
        y_moving_average = list()
        y_moving_average_standard_deviation = list()
        y_moving_average_counts = list()

        # Calculate the moving average for every list
        x = x_min
        while x <= x_max:
            # Construct a boolean area on which x values to use
            used_y_values = y_values[(x_values >= x - extend) & (x_values <= x + extend)]

            if len(used_y_values) >= 2:
                x_moving_average.append(x)
                y_moving_average.append(used_y_values.mean())
                y_moving_average_standard_deviation.append(numpy.std(used_y_values, ddof=1))
                y_moving_average_counts = len(used_y_values)

            x += x_step_size

        self.x_values = numpy.array(x_moving_average, dtype=numpy.float32)
        self.mean_values = numpy.array(y_moving_average, dtype=numpy.float32)
        self.standard_deviation_values = numpy.array(y_moving_average_standard_deviation, dtype=numpy.float32)
        self.counts_in_standard_deviation_values = numpy.array(y_moving_average_counts, dtype=numpy.uint16)

    def plot(self, axes: Axes, *, color="blue", linewidth=2, error_opacity=0.8, standard_error: bool = False, label="Moving average"):
        axes.plot(self.x_values, self.mean_values, color=color, linewidth=linewidth, label=label)

        error_bar_size = self.standard_deviation_values
        if standard_error:
            error_bar_size /= numpy.sqrt(self.counts_in_standard_deviation_values)

        axes.fill_between(self.x_values, self.mean_values - error_bar_size,
                          self.mean_values + error_bar_size, color=color, alpha=1 - error_opacity)


class LinesAverage(PlotAverage):
    """A moving average for a bunch of lines."""

    _lines: List[Tuple[List[float], List[float]]]
    _x_step_size: float

    def __init__(self, *lines: Tuple[List[float], List[float]], x_step_size: float = 1):
        """Creates the moving average. Each line is ([x1, x2, ...], [y1, y2, ...]), with the x in order from low to high."""
        self._lines = list(lines)
        self._x_step_size = x_step_size
        if x_step_size <= 0:
            raise ValueError(f"Illegal step size: {x_step_size}")

    def _get_min_max_x(self) -> Tuple[float, float]:
        """Gets the lowest and highest x values used in the lines. Returns (0, 1) if no lines are available."""
        min_x = None
        max_x = None
        for line_x, _ in self._lines:
            min_x = min_none(line_x[0], min_x)
            max_x = max_none(line_x[-1], max_x)
        if min_x is None:
            return 0, 1
        if max_x == min_x:
            return min_x, max_x + 1
        return min_x, max_x

    def _get_y_values_at(self, x: float) -> List[float]:
        """Returns a list of all y values of the lines at the given position,
        using linear interpolation if necessary. May return an empty list."""
        y_values = []
        for line_x, line_y in self._lines:
            if x <= line_x[0]:
                if x == line_x[0]:
                    y_values.append(line_y[0])
                continue
            if x >= line_x[-1]:
                if x == line_x[-1]:
                    y_values.append(line_y[-1])
                continue
            for i in range(len(line_x)):
                if x < line_x[i]:
                    # Need to grab y value in between point i and i-1

                    # i_fraction is 0 if we're at line_x[i] and 1 if at line_x[i - 1]
                    i_fraction = (line_x[i] - x) / (line_x[i] - line_x[i - 1])

                    y_value = i_fraction * line_y[i - 1] + (1 - i_fraction) * line_y[i]
                    y_values.append(y_value)
                    break
        return y_values

    def plot(self, axes: Axes, *, color="blue", linewidth=2, error_opacity=0.8, standard_error: bool = False, label="Moving average"):
        min_x, max_x = self._get_min_max_x()

        # Calculate error bounds
        x_error_values = list()
        y_error_values_min = list()
        y_error_values_mean = list()
        y_error_values_max = list()
        for x in numpy.arange(min_x + 0.01, max_x - 0.01, self._x_step_size):
            y_values = self._get_y_values_at(x)
            if len(y_values) <= 1:
                continue

            y_mean = numpy.mean(y_values)
            y_error = numpy.std(y_values, ddof=1)
            if standard_error:
                y_error /= numpy.sqrt(len(y_values))

            x_error_values.append(x)
            y_error_values_min.append(y_mean - y_error)
            y_error_values_mean.append(y_mean)
            y_error_values_max.append(y_mean + y_error)

        # Plot
        if len(x_error_values) > 0:
            axes.plot(x_error_values, y_error_values_mean, color=color, linewidth=linewidth, label=label)
            axes.fill_between(x_error_values, y_error_values_min,
                          y_error_values_max, color=color, alpha=1 - error_opacity)
