from typing import Dict, Any, List, Tuple

import numpy
from matplotlib.figure import Figure

from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.linking_analysis import linking_markers, cell_fate_finder
from organoid_tracker.linking_analysis.cell_fate_finder import CellFateType
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS, HISTOGRAM_RED, HISTOGRAM_BLUE, SANDER_GREEN, SANDER_RED, \
    SANDER_BLUE


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell deaths//Graph-Sister fate, position and time since last division...": lambda: _show_sisters_of_shed_cells(window)
    }

def _show_sisters_of_shed_cells(window: Window):
    experiment = window.get_experiment()
    resolution = experiment.images.resolution()

    # Records the (time since the last division, the axis position at death) for three different categories
    with_dead_sister = []
    with_dividing_sister = []
    with_idle_sister = []
    with_unknown_fate_h = []

    links = experiment.links
    position_data = experiment.position_data
    splines = experiment.splines
    for dead_cell in linking_markers.find_death_and_shed_positions(links, position_data):
        shedding_track = links.get_track(dead_cell)
        division = cell_division_finder.get_previous_division(links, dead_cell)
        if division is None:
            continue
        for daughter in division.daughters:
            if links.get_track(daughter) == shedding_track:
                continue
            # We now have two variables - the shedding_track and "daughter", which is now actually the
            # sister of shedding_track

            time_h = (dead_cell.time_point_number() - division.mother.time_point_number()) * resolution.time_point_interval_h
            other_fate =  cell_fate_finder.get_fate(experiment, daughter)
            axis_position_object = splines.to_position_on_original_axis(links, dead_cell)
            axis_position = axis_position_object.pos * resolution.pixel_size_x_um if axis_position_object is not None else 0

            if other_fate.type in cell_fate_finder.WILL_DIE_OR_SHED:
                with_dead_sister.append((time_h, axis_position))
            elif other_fate.type == CellFateType.WILL_DIVIDE:
                with_dividing_sister.append((time_h, axis_position))
            elif other_fate.type == CellFateType.JUST_MOVING:
                with_idle_sister.append((time_h, axis_position))
            else:
                with_unknown_fate_h.append((time_h, axis_position))
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(figure,
            with_dead_sister, with_dividing_sister, with_idle_sister, with_unknown_fate_h), size_cm=(14, 9))


def _draw_figure(figure: Figure, with_dead_sister: List[Tuple[float, float]],
                 with_dividing_sister: List[Tuple[float, float]], with_idle_sister: List[Tuple[float, float]],
                 with_unknown_sister_fate: List[Tuple[float, float]]):
    with_dead_sister = numpy.array(with_dead_sister, dtype=numpy.float64)
    with_dividing_sister = numpy.array(with_dividing_sister, dtype=numpy.float64)
    with_idle_sister = numpy.array(with_idle_sister, dtype=numpy.float64)
    with_unknown_sister_fate = numpy.array(with_unknown_sister_fate, dtype=numpy.float64)

    axes = figure.gca()
    axes.scatter(x=with_dead_sister[:, 0], y=with_dead_sister[:, 1],
                 color=SANDER_BLUE, marker="X", label="Dying or dead sister")
    axes.scatter(x=with_dividing_sister[:, 0], y=with_dividing_sister[:, 1],
                 color=SANDER_RED, marker="P", label="Dividing sister")
    axes.scatter(x=with_idle_sister[:, 0], y=with_idle_sister[:, 1],
                 color=SANDER_GREEN, marker="o", label="Idle sister")
    axes.scatter(x=with_unknown_sister_fate[:, 0], y=with_unknown_sister_fate[:, 1],
                 color=(0.3, 0.3, 0.3), marker="s", label="Unknown sister fate")

    axes.set_xlabel("Time since last division (h)")
    axes.set_ylabel("Crypt-villus axis position (µm)")
    axes.legend()
    figure.tight_layout()
