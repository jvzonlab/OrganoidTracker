from typing import Dict, Any, Tuple, Optional

import numpy
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray

from ai_track.core import UserError
from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.core.score import Family
from ai_track.gui import dialog
from ai_track.gui.window import Window
from ai_track.linking import cell_division_finder
from ai_track.linking_analysis import cell_fate_finder, particle_age_finder
from ai_track.linking_analysis.cell_fate_finder import CellFateType
from ai_track.util.moving_average import MovingAverage


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell cycle//Cycle lengths of two generations...": lambda: _view_cell_cycle_length(window)
    }


def _view_cell_cycle_length(window: Window):
    experiment = window.get_experiment()
    links = experiment.links
    if not links.has_links():
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")

    third_variable_getter = _ThirdVar()
    try:
        time_point_duration_h = experiment.images.resolution().time_point_interval_m / 60
    except ValueError:
        raise UserError("No resolution set", "The resolution of the images was not set. Cannot plot anything.")

    dialog.popup_figure(window.get_gui_experiment(), lambda fig: _draw_cell_cycle_length(fig, links, time_point_duration_h,
                                                                             third_variable_getter))


class _ThirdVar:
    def get_number(self, daughter: Position, next_division: Family):
        return 1

    def show_average(self) -> bool:
        return True

    def get_cmap(self) -> str:
        return "Greys"

    def get_colobar_label(self) -> Optional[str]:
        """If this is None, then no color or color bar is used: the the Third Variable will essentiaally be ignored."""
        return None


class _CellFateVar(_ThirdVar):
    experiment: Experiment

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def get_number(self, daughter: Position, next_division: Family) -> float:
        combined_fate = None
        for next in next_division.daughters:
            cell_fate = cell_fate_finder.get_fate(self.experiment, next).type
            if combined_fate is None:
                combined_fate = cell_fate
                continue
            if combined_fate == cell_fate:
                continue
            combined_fate = CellFateType.UNKNOWN

        if combined_fate == CellFateType.WILL_DIVIDE:
            return 1
        elif combined_fate == CellFateType.JUST_MOVING or combined_fate == CellFateType.WILL_DIE\
                or combined_fate == CellFateType.WILL_SHED:
            return 0
        else:
            return 0.5

    def show_average(self) -> bool:
        return False

    def get_colobar_label(self) -> str:
        return "Cell fate after next division (1 = divides)"


def _draw_cell_cycle_length(figure: Figure, links: Links, time_point_duration_h: float,
                            third_variable_getter: _ThirdVar):
    previous_cycle_durations = list()
    cycle_durations = list()
    third_variables = list()  # Used for color, can be z position

    # Find all families and their next division
    for family in cell_division_finder.find_families(links):
        previous_cycle_duration = particle_age_finder.get_age(links, family.mother)
        if previous_cycle_duration is None:
            continue

        division_time = family.mother.time_point_number()
        for daughter in family.daughters:
            next_division = cell_division_finder.get_next_division(links, daughter)
            if next_division is None:
                continue
            cycle_duration = next_division.mother.time_point_number() - division_time
            previous_cycle_durations.append(previous_cycle_duration)
            cycle_durations.append(cycle_duration)
            third_variables.append(third_variable_getter.get_number(daughter, next_division))

    if len(cycle_durations) == 0:
        raise UserError("No cell cycles found", "The linking data contains no sequences of two complete cell cycles. "
                                                "Therefore, we cannot plot anything.")

    # Convert to numpy, get statistics
    previous_cycle_durations = numpy.array(previous_cycle_durations, dtype=numpy.int32) * time_point_duration_h
    cycle_durations = numpy.array(cycle_durations, dtype=numpy.int32) * time_point_duration_h
    third_variables = numpy.array(third_variables, dtype=numpy.float32)
    plot_start = min(cycle_durations.min(), previous_cycle_durations.min()) / 2
    plot_limit = max(previous_cycle_durations.max(), cycle_durations.max()) * 1.1

    window_size = 11
    moving_average = MovingAverage(previous_cycle_durations, cycle_durations, window_width=window_size)

    axes = figure.gca()
    axes.plot(numpy.arange(plot_start, plot_limit), numpy.arange(plot_start, plot_limit), color="orange",
              label="Equal durations line")
    if third_variable_getter.show_average():
        moving_average.plot(axes, label=f"Moving average ({window_size} time points)")

    if third_variable_getter.get_colobar_label() is not None:
        scatterplot = axes.scatter(x=previous_cycle_durations, y=cycle_durations, c=third_variables, s=25, lw=1,
                                   cmap=third_variable_getter.get_cmap(), edgecolors="black")
        divider = make_axes_locatable(axes)
        axes_on_right = divider.append_axes("right", size="5%", pad=0.1)
        figure.colorbar(scatterplot, cax=axes_on_right).set_label(third_variable_getter.get_colobar_label())
    else:
        axes.scatter(x=previous_cycle_durations, y=cycle_durations, c="black", s=9, lw=0, alpha=0.7)
    axes.set_xlim(plot_start, plot_limit)
    axes.set_ylim(plot_start, plot_limit)
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlabel("Duration of cell cycle of mother (h)")
    axes.set_ylabel("Duration of cell cycle of daughter (h)")
    axes.legend(loc="lower right")
