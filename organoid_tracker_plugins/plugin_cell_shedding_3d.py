# This import registers the 3D projection, but is otherwise unused.
from typing import Dict, Any, List

import numpy
from matplotlib.figure import Figure
import numpy as np

from organoid_tracker.coordinate_system.spherical_coordinates import SphericalCoordinate
from organoid_tracker.core import UserError
from organoid_tracker.core.beacon_collection import ClosestBeacon
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell deaths//3D-Projection of cell shedding locations...": lambda: _open(window)
    }


def _open(window: Window):
    experiment = window.get_experiment()
    beacons = experiment.beacons
    resolution = experiment.images.resolution()
    closest_beacon_to_shed_positions = list()

    for shed_cell in linking_markers.find_shed_positions(experiment.links, experiment.position_data):
        closest_beacon = beacons.find_closest_beacon(shed_cell, resolution)
        if closest_beacon is None:
            raise UserError("No beacon found", f"No beacon was found at time point {shed_cell.time_point_number()}."
                            f" Beacons are used to mark the center of the villus. Make sure to add a beacon at this"
                            f" time point, so that we know the relative position of the shed cell.")
        closest_beacon_to_shed_positions.append(closest_beacon)

    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _plot(figure, closest_beacon_to_shed_positions))


def _plot(fig: Figure, closest_beacon_to_shed_positions: List[ClosestBeacon]):
    from mpl_toolkits.mplot3d import Axes3D
    from organoid_tracker.util.mpl_helper_3d import draw_sphere

    np.random.seed(19680801)
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    radius_list = list()
    x_list = list()
    y_list = list()
    z_list = list()
    for closest_beacon in closest_beacon_to_shed_positions:
        radius_list.append(closest_beacon.distance_um)
        difference = closest_beacon.difference_um()
        projected = SphericalCoordinate.from_cartesian(difference).to_cartesian(radius=1)
        x_list.append(projected.x)
        y_list.append(projected.y)
        z_list.append(projected.z)

    draw_sphere(ax, color=(1, 0, 0, 0.2), radius=0.9)
    ax.scatter(x_list, y_list, z_list, marker='o', color="black")

    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_zticks([-1, 1])
    ax.set_xticklabels(["-", "+"])
    ax.set_yticklabels(["-", "+"])
    ax.set_zticklabels(["-", "+"])
    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])
    ax.set_zlim([1, -1])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

