from typing import Tuple, List, Dict, Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Over space and time-Number of cells over time...": lambda: _view_cell_count_over_time(window),
    }


def _view_cell_count_over_time(window: Window):
    experiment = window.get_experiment()
    positions = experiment.positions
    resolution = experiment.images.resolution()

    time_list = list()
    count_list = list()
    for time_point in positions.time_points():
        time_list.append(resolution.time_point_interval_h * time_point.time_point_number())
        count_list.append(len(positions.of_time_point(time_point)))
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _plot(figure, time_list, count_list))


def _plot(figure: Figure, time_list: List[float], count_list: List[int]):
    axes: Axes = figure.gca()
    axes.plot(time_list, count_list, color=SANDER_APPROVED_COLORS[0])
    axes.set_xlabel("Time (h)")
    axes.set_ylabel("Number of cells")
