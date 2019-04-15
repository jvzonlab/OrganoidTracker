from typing import List, Union

import numpy
from matplotlib.axes import Axes
from numpy import ndarray


class MovingAverage:
    """A moving average calculation. Usage example:

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
    standard_error_values: ndarray  # Standard error in the mean. Useful for plt.fill_between.

    def __init__(self, x_values: Union[List[float], ndarray], y_values: Union[List[float], ndarray], *,
                 window_width: float, step_size: float = 1):
        """Calculates the moving average with the given width. The resolution determines the step size in which the
        average is moved over the data. The lower the value, the longer the calculation takes."""
        if step_size <= 0:
            raise ValueError(f"resolution must be positive, but was {step_size}")
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
        y_moving_average_standard_error = list()

        # Calculate the moving average for every list
        x = x_min
        while x <= x_max:
            # Construct a boolean area on which x values to use
            used_y_values = y_values[(x_values >= x - extend) & (x_values <= x + extend)]

            if len(used_y_values) >= 2:
                x_moving_average.append(x)
                y_moving_average.append(used_y_values.mean())
                y_moving_average_standard_error.append(numpy.std(used_y_values, ddof=1) / numpy.sqrt(len(used_y_values)))

            x += step_size

        self.x_values = numpy.array(x_moving_average, dtype=numpy.float32)
        self.mean_values = numpy.array(y_moving_average, dtype=numpy.float32)
        self.standard_error_values = numpy.array(y_moving_average_standard_error, dtype=numpy.float32)

    def plot(self, axes: Axes, *, color="blue", linewidth=2, error_alpha=0.2, label="Moving average"):
        """Plots the moving average to the given axis. For example:

            plot(plt.gca(), label="My moving average")
        """
        axes.plot(self.x_values, self.mean_values, color=color, linewidth=linewidth, label=label)
        axes.fill_between(self.x_values, self.mean_values - self.standard_error_values,
                          self.mean_values + self.standard_error_values, color=color, alpha=error_alpha)
