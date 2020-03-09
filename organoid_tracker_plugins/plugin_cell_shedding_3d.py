# This import registers the 3D projection, but is otherwise unused.
from typing import Dict, Any, List

from matplotlib.figure import Figure

from organoid_tracker.coordinate_system.spherical_coordinates import SphericalCoordinate
from organoid_tracker.core import UserError, TimePoint
from organoid_tracker.core.beacon_collection import ClosestBeacon
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell deaths//3D-Projection of cell shedding locations...": lambda: _open(window)
    }


def _open(window: Window):
    experiment = window.get_experiment()
    closest_beacon_to_shed_positions = _get_closest_beacon_to_shed_positions(experiment)
    closest_beacon_to_crypt_bottoms = _get_closest_beacon_to_crypt_bottoms(experiment)

    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _plot(figure, closest_beacon_to_shed_positions,
                                                                          closest_beacon_to_crypt_bottoms))


def _get_closest_beacon_to_shed_positions(experiment: Experiment) -> List[ClosestBeacon]:
    """Gets the relative position to the closest beacon for every shed cell."""
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
    return closest_beacon_to_shed_positions


def _get_closest_beacon_to_crypt_bottoms(experiment: Experiment) -> List[ClosestBeacon]:
    beacons = experiment.beacons
    resolution = experiment.images.resolution()
    closest_beacon_to_crypt_bottoms = list()
    last_time_point = TimePoint(experiment.splines.last_time_point_number())
    for spline_index, spline in experiment.splines.of_time_point(last_time_point):
        crypt_bottom = spline.from_position_on_axis(0)
        if crypt_bottom is not None:
            crypt_bottom_position = Position(crypt_bottom[0], crypt_bottom[1], spline.get_z(), time_point=last_time_point)
            result = beacons.find_closest_beacon(crypt_bottom_position, resolution)
            closest_beacon_to_crypt_bottoms.append(result)
    return closest_beacon_to_crypt_bottoms


def _plot(fig: Figure, closest_beacon_to_shed_positions: List[ClosestBeacon], closest_beacon_to_crypt_bottom: List[ClosestBeacon]):
    from mpl_toolkits.mplot3d import Axes3D
    from organoid_tracker.util.mpl_helper_3d import draw_sphere
    # Right now, this ignores the fact that there may be multiple beacons, and just plots everything on a single sphere

    # noinspection PyTypeChecker
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    draw_sphere(ax, color=(1, 0, 0, 0.2), radius=0.9)

    shedding_x_list, shedding_y_list, shedding_z_list = list(), list(), list()
    for closest_beacon in closest_beacon_to_shed_positions:
        difference = closest_beacon.difference_um()
        projected = SphericalCoordinate.from_cartesian(difference).to_cartesian(radius=1)
        shedding_x_list.append(projected.x)
        shedding_y_list.append(projected.y)
        shedding_z_list.append(projected.z)
    ax.scatter(shedding_x_list, shedding_y_list, shedding_z_list, marker='o', color="black")

    crypt_x1_list, crypt_y1_list, crypt_z1_list = list(), list(), list()
    crypt_x2_list, crypt_y2_list, crypt_z2_list = list(), list(), list()
    for closest_beacon in closest_beacon_to_crypt_bottom:
        difference = closest_beacon.difference_um()
        projected_nearby = SphericalCoordinate.from_cartesian(difference).to_cartesian(radius=0.9)
        projected_further_away = SphericalCoordinate.from_cartesian(difference).to_cartesian(radius=1.8)
        ax.plot([projected_nearby.x, projected_further_away.x],
                [projected_nearby.y, projected_further_away.y],
                [projected_nearby.z, projected_further_away.z], linewidth=3, color="red")

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

