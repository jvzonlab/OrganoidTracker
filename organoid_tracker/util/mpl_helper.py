from typing import Tuple, Optional, Union, List

import numpy
from matplotlib import colors, cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize

# Colors Sander Tans likes in his figures - these are all bright colors
# Source: flatuicolors.com - US palette
from numpy import ndarray

from organoid_tracker.core.typing import MPLColor

SANDER_APPROVED_COLORS = colors.to_rgba_array({"#55efc4", "#fdcb6e", "#636e72", "#74b9ff", "#e84393",
                                               "#ff7675", "#fd79a8", "#fab1a0", "#e84393"})

# A bit darker colors
SANDER_BLUE = "#0984e3"
SANDER_RED = "#d63031"
SANDER_GREEN = "#00b894"

# Overlapping colors for histogram
HISTOGRAM_RED = (1, 0, 0, 0.7)
HISTOGRAM_BLUE = (0, 0, 1, 0.5)

def _create_qualitative_colormap() -> ListedColormap:
    color_list = cm.get_cmap('jet', 256)(numpy.linspace(0, 1, 256))
    numpy.random.shuffle(color_list)
    color_list[0] = numpy.array([0, 0, 0, 1])  # First color is black
    return ListedColormap(color_list)


QUALITATIVE_COLORMAP = _create_qualitative_colormap()


def line_infinite(ax: Axes, x1: float, y1: float, x2: float, y2: float, color: MPLColor="red", linewidth: int = 2, alpha: float = 1):
    """Draws a 2D line that extends a long while across the current view."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Two cases in which we can have truly infinite lines
    if dx == 0:
        ax.axvline(x1, color=color, linewidth=linewidth, alpha=alpha)
        return
    if dy == 0:
        ax.axhline(y1, color=color, linewidth=linewidth, alpha=alpha)
        return

    # Calculate scale
    axes = ax.axis()
    axes_dx = abs(axes[0] - axes[1])
    axes_dy = abs(axes[2] - axes[3])
    scale = 1000 * max(axes_dx, axes_dy) / max(dx, dy)

    # Plot the line
    ax.plot([x1, x1 + scale * (x2 - x1)], [y1, y1 + scale * (y2 - y1)], color=color, linewidth=linewidth, alpha=alpha)
    ax.plot([x1, x1 - scale * (x2 - x1)], [y1, y1 - scale * (y2 - y1)], color=color, linewidth=linewidth, alpha=alpha)
    ax.set_xlim([axes[0], axes[1]])
    ax.set_ylim([axes[2], axes[3]])


AxesLimits = Tuple[Tuple[float, float], Tuple[float, float]]  # Type definition
def store_axes_limits(axes: Axes) -> Optional[AxesLimits]:
    """Returns the current limits of the axes. Returns None if they have their default value."""
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    if abs(xlim[1] - xlim[0]) <= 1:
        return None
    return xlim, ylim


def restore_axes_limits(axes: Axes, limits: Optional[AxesLimits]):
    """Restores the current limits of the axis. Does nothing if limits is None."""
    if limits is None:
        return
    x_lim, y_lim = limits
    axes.set_xlim(*x_lim)
    axes.set_ylim(*y_lim)


def plot_multicolor(ax: Axes, x: Union[List[float], ndarray], y: Union[List[float], ndarray],
                    *, colors: ndarray, linewidth: float = 2) -> LineCollection:
    """Plots a multicolored line. The array named c must contain numbers between 0 and 1."""

    # Based on https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    lc = LineCollection(segments, colors=colors)
    lc.set_linewidth(linewidth)
    return ax.add_collection(lc)
