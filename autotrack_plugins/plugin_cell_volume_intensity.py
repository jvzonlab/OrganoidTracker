from statistics import median
from typing import Optional, Tuple, List, Callable, Dict, Any

from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.core.links import Links
from autotrack.core.position import Position
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking import cell_division_finder

GetStatistic = Callable[[Experiment, Position], float]  # A function that gets some statistic of a cell, like its volume
PointList = Tuple[List[float], List[float]]  # A list of x values and a list of y values


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Over time-Cell volumes over time...": lambda: _show_cell_volumes(window),
        "Graph//Over time-Cell intensities over time...": lambda: _show_cell_intensities(window)
    }


def _show_cell_volumes(window: Window):
    def draw(figure: Figure):
        _plot_volumes(window.get_experiment(), figure)

    dialog.popup_figure(window.get_gui_experiment(), draw)


def _show_cell_intensities(window: Window):
    def draw(figure: Figure):
        _plot_intensities(window.get_experiment(), figure)

    dialog.popup_figure(window.get_gui_experiment(), draw)


def _get_volume(experiment: Experiment, position: Position) -> Optional[float]:
    shape = experiment.positions.get_shape(position)
    try:
        return shape.volume()
    except NotImplementedError:
        return None


def _get_intensity(experiment: Experiment, position: Position) -> Optional[float]:
    shape = experiment.positions.get_shape(position)
    try:
        return shape.intensity()
    except ValueError:
        return None


def _plot_volumes(experiment: Experiment, figure: Figure, mi_start=0, line_count=300, starting_time_point=50):
    """Plots the volumes of all cells in time. T=0 represents a cell division."""
    _plot_mother_stat(experiment, figure, _get_volume, 'Cell volume (px$^3$)', mi_start, line_count,
                      starting_time_point)


def _plot_mother_stat(experiment: Experiment, figure: Figure, stat: GetStatistic, y_label: str, mi_start: int,
                      line_count: int, starting_time_point: int):
    links = experiment.links
    if not links.has_links():
        raise UserError("No cell links", "No cell links were loaded, so we cannot track cell statistics over time.")
    mothers = [mother for mother in cell_division_finder.find_mothers(links) if mother.time_point_number() >= starting_time_point]
    mothers = mothers[mi_start:mi_start + line_count]
    axes = figure.gca()
    show_legend = line_count <= 5

    all_values = []
    lines = []
    for mother in mothers:
        time_point_numbers, volumes = _data_into_past_until_division(experiment, mother, links, stat)
        color = None if show_legend else (0, 0, 0, 0.2)
        lines.append(axes.plot(time_point_numbers, volumes, label=str(mother), color=color))
        all_values += volumes
    if len(all_values) == 0:
        raise UserError("No data to display", "No cell statistics were recorded. These data normally come from a "
                                              "Gaussian fit. Did you perform such a fit on the data?")
    axes.set_ylim(bottom=0, top=median(all_values) * 2)
    axes.set_xlabel('Time point')
    axes.set_ylabel(y_label)
    if show_legend:
        axes.legend()


def _plot_intensities(experiment: Experiment, figure: Figure, mi_start=0, line_count=300, starting_time_point=50):
    """Plots the intensities of all cells in time. T=0 represents a cell division."""
    _plot_mother_stat(experiment, figure, _get_intensity, 'Cell intensity (A.U.)', mi_start, line_count,
                      starting_time_point)


def _data_into_past_until_division(experiment: Experiment, starting_point: Position, links: Links,
                                   func: GetStatistic) -> PointList:
    position = starting_point
    x_values = []
    y_values = []
    while position is not None:
        y_value = func(experiment, position)
        if y_value is not None:
            x_values.append(position.time_point_number() - starting_point.time_point_number())
            y_values.append(y_value)

        position = _get_previous(position, links)
    return x_values, y_values


def _get_previous(position: Position, links: Links) -> Optional[Position]:

    # Find the single previous position
    previous_positions = links.find_pasts(position)
    if len(previous_positions) != 1:
        return None
    previous = previous_positions.pop()

    # Find the single next position of the previous (ensures that we are not doing another cell division)
    next_positions = links.find_futures(previous)
    if len(next_positions) != 1:
        return None  # This is a mother cell, so don't take it into account

    return previous

