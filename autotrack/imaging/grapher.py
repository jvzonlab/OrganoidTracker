from typing import List, Union

import numpy as np
import matplotlib.path as mpath
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize
from numpy import ndarray


_FloatList = Union[ndarray, List[float]]


def colorline(
    x: _FloatList, y: _FloatList, z=None, cmap: Colormap = cm.get_cmap('copper'), norm: Normalize = Normalize(0.0, 1.0),
        linewidth: int = 1, alpha: float = 1.0) -> LineCollection:
    """
    From https://stackoverflow.com/a/25941474

    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = _make_segments(x, y)
    return LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
