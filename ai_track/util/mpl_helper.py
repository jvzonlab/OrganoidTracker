from typing import Tuple, Optional

import numpy
from matplotlib import colors, cm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

# Colors Sander Tans likes in his figures - these are all bright colors
# Source: flatuicolors.com - US palette
from ai_track.core.typing import MPLColor

SANDER_APPROVED_COLORS = colors.to_rgba_array({"#55efc4", "#fdcb6e", "#636e72", "#74b9ff", "#e84393",
                                               "#ff7675", "#fd79a8", "#fab1a0", "#e84393"})

# A bit darker colors
BAR_COLOR_1 = "#0984e3"
BAR_COLOR_2 = "#d63031"

# Overlapping colors for histogram
HISTOGRAM_RED = (1, 0, 0, 0.7)
HISTOGRAM_BLUE = (0, 0, 1, 0.5)


def _create_qualitative_colormap() -> ListedColormap:
    color_list = cm.get_cmap('jet', 256)(numpy.linspace(0, 1, 256))
    numpy.random.shuffle(color_list)
    color_list[0] = numpy.array([0, 0, 0, 1])  # First color is black
    return ListedColormap(color_list)


QUALITATIVE_COLORMAP = _create_qualitative_colormap()


def line_infinite(ax: Axes, x1: float, y1: float, x2: float, y2: float, color: MPLColor="red", linewidth: int = 2):
    """Draws a 2D line that extends a long while across the current view."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Two cases in which we can have truly infinite lines
    if dx == 0:
        ax.axvline(x1, color=color, linewidth=linewidth)
        return
    if dy == 0:
        ax.axhline(y1, color=color, linewidth=linewidth)
        return

    # Calculate scale
    axes = ax.axis()
    axes_dx = abs(axes[0] - axes[1])
    axes_dy = abs(axes[2] - axes[3])
    scale = 1000 * max(axes_dx, axes_dy) / max(dx, dy)

    # Plot the line
    ax.plot([x1, x1 + scale * (x2 - x1)], [y1, y1 + scale * (y2 - y1)], color=color, linewidth=linewidth)
    ax.plot([x1, x1 - scale * (x2 - x1)], [y1, y1 - scale * (y2 - y1)], color=color, linewidth=linewidth)
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
