import numpy
from mpl_toolkits.mplot3d import Axes3D

from organoid_tracker.core.typing import MPLColor


def draw_sphere(ax: Axes3D, *, radius: float = 1, color: MPLColor = "red"):
    """Draws a sphere."""
    u, v = numpy.mgrid[0:2 * numpy.pi:20j, 0:numpy.pi:10j]
    x = (numpy.cos(u) * numpy.sin(v)) * radius
    y = (numpy.sin(u) * numpy.sin(v)) * radius
    z = numpy.cos(v) * radius
    ax.plot_surface(x, y, z, color=color)
