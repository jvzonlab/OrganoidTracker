import numpy
from mpl_toolkits.mplot3d import Axes3D

from organoid_tracker.core.typing import MPLColor
from organoid_tracker.core.vector import Vector3


def draw_sphere(ax: Axes3D, *, center: Vector3, radius: float = 1, color: MPLColor = "red"):
    """Draws a sphere."""
    u, v = numpy.mgrid[0:2 * numpy.pi:20j, 0:numpy.pi:10j]
    x = (numpy.cos(u) * numpy.sin(v)) * radius + center.x
    y = (numpy.sin(u) * numpy.sin(v)) * radius + center.y
    z = numpy.cos(v) * radius + center.z
    ax.plot_surface(x, y, z, color=color)
