from typing import List, Union

import numpy
from matplotlib.axes import Axes
from numpy import ndarray


class MovingAverage:
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
        """Plots the moving average to the given axis. For example:

            plot(plt.gca(), label="My moving average")

        The error bar is either the standard deviation (if standard_error is False) or the standard error (if standard_error is True).
        """
        axes.plot(self.x_values, self.mean_values, color=color, linewidth=linewidth, label=label)

        error_bar_size = self.standard_deviation_values
        if standard_error:
            error_bar_size /= numpy.sqrt(self.counts_in_standard_deviation_values)

        axes.fill_between(self.x_values, self.mean_values - error_bar_size,
                          self.mean_values + error_bar_size, color=color, alpha=1 - error_opacity)
