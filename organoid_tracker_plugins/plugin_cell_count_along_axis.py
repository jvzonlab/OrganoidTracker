from typing import Dict, Any, List, Optional

from matplotlib.figure import Figure

from organoid_tracker.core import UserError, TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Over space and time-Histogram of cells on crypt axis...": lambda: _view_crypt_axis_positions(window),
    }


def _view_crypt_axis_positions(window: Window):
    experiment = window.get_experiment()

    min_time_point_number = experiment.first_time_point_number()
    max_time_point_number = experiment.last_time_point_number()
    if min_time_point_number is None or max_time_point_number is None:
        raise UserError("No time points loaded", "Cannot plot anything, as there are no time points loaded.")
    time_point_number = dialog.prompt_int("Time point", f"Which time point do you want to view?"
                                          f" ({min_time_point_number} - {max_time_point_number}, inclusive)",
                                          minimum=min_time_point_number, maximum=max_time_point_number)
    if time_point_number is None:
        return  # Cancelled
    time_point = experiment.get_time_point(time_point_number)

    data = _get_crypt_axis_positions(experiment, time_point)
    if len(data) == 0:
        raise UserError("No data found", "No date found. Are the cell positions defined?"
                                         " Did you draw crypt-villus axes at every time point?")
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(figure, data))


def _get_crypt_axis_positions(experiment: Experiment, time_point: Optional[TimePoint]) -> Dict[int, List[float]]:
    """Gets all used crypt axis positions."""
    data_axes = experiment.splines
    links = experiment.links

    return_value = dict()
    positions = experiment.positions if time_point is None else experiment.positions.of_time_point(time_point)
    resolution = experiment.images.resolution()
    for position in positions:
        axis_position = data_axes.to_position_on_original_axis(links, position)
        if axis_position is not None:
            if axis_position.axis_id not in return_value:
                return_value[axis_position.axis_id] = []
            return_value[axis_position.axis_id].append(axis_position.pos * resolution.pixel_size_x_um)

    return return_value


def _draw_figure(figure: Figure, data: Dict[int, List[float]]):
    axes = figure.gca()
    for axis_id, axis_values in data.items():
        axes.hist(axis_values, label="Axis " + str(axis_id), alpha=0.5)
    if len(data) > 1:
        axes.legend()
    axes.set_xlabel("Axis position (μm)")
    axes.set_ylabel("Amount")
