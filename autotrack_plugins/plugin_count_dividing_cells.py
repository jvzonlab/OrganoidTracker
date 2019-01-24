from typing import List

from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking_analysis import cell_fate_finder, linking_markers
from autotrack.linking_analysis.cell_fate_finder import CellFate, CellFateType


def get_menu_items(window: Window):
    # This function is automatically called for any file named plugin_ ... .py in the plugins folder
    # You need to return a dictionary of menu options here
    return {
        "Graph//Cell cycle-Number of dividing cells...":
            lambda: _show_number_of_dividing_cells(window)
    }


def _show_number_of_dividing_cells(window: Window):
    """Shows the number of dividing cells."""
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("No linking data", "No linking data is found. Cannot plot anything.")
    resolution = experiment.images.resolution()

    time_point_hours = []
    dividing_counts_min = []
    dividing_counts_max = []
    paneth_counts = []
    total_counts = []

    for time_point in experiment.time_points():
        dividing_count_min = 0
        dividing_count_max = 0
        paneth_count = 0
        total_count = 0
        for position in experiment.positions.of_time_point(time_point):
            fate = cell_fate_finder.get_fate(experiment, position)
            if fate.type == CellFateType.WILL_DIVIDE:
                dividing_count_min += 1
                dividing_count_max += 1
            elif fate.type == CellFateType.UNKNOWN:
                dividing_count_max += 1  # Could be dividing, but we're not sure

            cell_type = linking_markers.get_position_type(experiment.links, position)
            if cell_type == "PANETH":
                paneth_count += 1

            total_count += 1

        time_point_hours.append(time_point.time_point_number() * resolution.time_point_interval_h)
        dividing_counts_min.append(dividing_count_min)
        dividing_counts_max.append(dividing_count_max)
        paneth_counts.append(paneth_count)
        total_counts.append(total_count)

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(figure, time_point_hours,
                                                                                 dividing_counts_min,
                                                                                 dividing_counts_max,
                                                                                 paneth_counts,
                                                                                 total_counts))


def _draw_figure(figure: Figure, time_point_hours: List[int], dividing_counts_min: List[int],
                 dividing_counts_max: List[int], paneth_counts: List[int], total_counts: List[int]):
    axes = figure.gca()
    axes.set_title("Number of dividing cells over time")
    axes.plot(time_point_hours, dividing_counts_min, label="Dividing cells")
    axes.fill_between(time_point_hours, dividing_counts_min, dividing_counts_max)
    axes.plot(time_point_hours, paneth_counts, color="red", label="Paneth cells")
    axes.plot(time_point_hours, total_counts, color="black", label="All cells")
    axes.set_ylabel("Number of cells (min and max)")
    axes.set_xlabel("Time (h)")
    axes.legend()
