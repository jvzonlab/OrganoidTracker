from typing import List, Dict, Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from organoid_tracker.core import Color
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.linking_analysis import cell_fate_finder
from organoid_tracker.linking_analysis.cell_fate_finder import CellFateType


def get_menu_items(window: Window):
    # This function is automatically called for any file named plugin_ ... .py in the plugins folder
    # You need to return a dictionary of menu options here
    return {
        "Graph//Cell cycle-Cell cycle//Number of dividing cells over time...":
            lambda: _show_number_of_dividing_cells(window)
    }


def _show_number_of_dividing_cells(window: Window):
    """Shows the number of dividing cells."""
    dividing_cells = []
    for experiment in window.get_active_experiments():
        dividing_cells.append(_DividingCells(experiment))

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(figure, dividing_cells),
                        export_function=lambda: _export_figure(dividing_cells))


class _DividingCells:
    experiment_color: Color
    time_point_hours: List[float]
    dividing_counts_min: List[int]
    dividing_counts_max: List[int]
    paneth_counts: List[int]
    total_counts: List[int]

    def __init__(self, experiment: Experiment):
        self.experiment_color = experiment.color
        self.time_point_hours = []
        self.dividing_counts_min = []
        self.dividing_counts_max = []
        self.paneth_counts = []
        self.total_counts = []

        resolution = experiment.images.resolution()
        for time_point in experiment.time_points():
            if time_point.time_point_number() >= \
                    experiment.last_time_point_number() - experiment.division_lookahead_time_points:
                continue
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

                cell_type = position_markers.get_position_type(experiment.positions, position)
                if cell_type == "PANETH":
                    paneth_count += 1

                total_count += 1

            self.time_point_hours.append(time_point.time_point_number() * resolution.time_point_interval_h)
            self.dividing_counts_min.append(dividing_count_min)
            self.dividing_counts_max.append(dividing_count_max)
            self.paneth_counts.append(paneth_count)
            self.total_counts.append(total_count)

    def to_dictionary(self) -> Dict[str, Any]:
        return {
            "experiment_color": self.experiment_color.to_rgb_floats(),
            "time_point_hours": self.time_point_hours,
            "dividing_counts_min": self.dividing_counts_min,
            "dividing_counts_max": self.dividing_counts_max,
            "paneth_counts": self.paneth_counts,
            "total_counts": self.total_counts
        }

def _draw_figure(figure: Figure, dividing_cells: List[_DividingCells]):
    axes: Axes = figure.gca()
    axes.set_title("Number of dividing cells over time")
    if len(dividing_cells) == 1:
        single_experiment = dividing_cells[0]
        axes.plot(single_experiment.time_point_hours, single_experiment.dividing_counts_min, label="Dividing cells")
        axes.fill_between(single_experiment.time_point_hours, single_experiment.dividing_counts_min,
                          single_experiment.dividing_counts_max)
        axes.plot(single_experiment.time_point_hours, single_experiment.paneth_counts, color="red",
                  label="Paneth cells")
        axes.plot(single_experiment.time_point_hours, single_experiment.total_counts, color="black", label="All cells")
        axes.set_ylabel("Number of cells (min and max)")
        axes.legend()
    else:
        for single_experiment in dividing_cells:
            color = single_experiment.experiment_color.to_rgba_floats()
            axes.plot(single_experiment.time_point_hours, single_experiment.dividing_counts_min, color=color,
                      linewidth=1.65)
        axes.set_ylabel("Number of dividing cells")
    axes.set_xlabel("Time (h)")


def _export_figure(dividing_cells: List[_DividingCells]) -> Dict[str, Any]:
    return {
        "experiments": [d.to_dictionary() for d in dividing_cells]
    }

