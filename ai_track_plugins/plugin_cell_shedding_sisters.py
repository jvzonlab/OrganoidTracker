from typing import Dict, Any

from matplotlib.figure import Figure

from ai_track.gui import dialog
from ai_track.gui.window import Window
from ai_track.linking import cell_division_finder
from ai_track.linking_analysis import linking_markers, cell_fate_finder
from ai_track.linking_analysis.cell_fate_finder import CellFateType
from ai_track.util.mpl_helper import SANDER_APPROVED_COLORS, HISTOGRAM_RED, HISTOGRAM_BLUE


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell deaths//Graph-Time since last division...": lambda: _show_sisters_of_shed_cells(window)
    }

def _show_sisters_of_shed_cells(window: Window):
    experiment = window.get_experiment()
    resolution = experiment.images.resolution()

    # Records the times since the last division for three different categories
    times_with_symmetric_fate_h = []
    times_with_asymmetric_fate_h = []
    times_with_unknown_fate_h = []

    links = experiment.links
    for shed_cell in linking_markers.find_shed_positions(links):
        shedding_track = links.get_track(shed_cell)
        division = cell_division_finder.get_previous_division(links, shed_cell)
        if division is None:
            continue
        for daughter in division.daughters:
            if links.get_track(daughter) == shedding_track:
                continue
            # We now have two variables - the shedding_track and "daughter", which is now actually the
            # sister of shedding_track

            time_h = (shed_cell.time_point_number() - division.mother.time_point_number()) * resolution.time_point_interval_h
            other_fate =  cell_fate_finder.get_fate(experiment, daughter)

            if other_fate.type == CellFateType.UNKNOWN:
                times_with_unknown_fate_h.append(time_h)
            elif other_fate.type not in cell_fate_finder.WILL_DIE_OR_SHED:
                print("Asymmetric fate at", time_h, "h after division: other is", other_fate.type)
                times_with_asymmetric_fate_h.append(time_h)
            else:
                times_with_symmetric_fate_h.append(time_h)
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_figure(figure,
            times_with_symmetric_fate_h, times_with_asymmetric_fate_h, times_with_unknown_fate_h), size_cm=(7,5))


def _draw_figure(figure: Figure, times_with_symmetric_fate_h, times_with_asymmetric_fate_h, times_with_unknown_fate_h):
    axes = figure.gca()
    axes.scatter(x=times_with_symmetric_fate_h, y=[0]*len(times_with_symmetric_fate_h),
                 color=HISTOGRAM_RED, marker="^", label="Symmetric fate")
    axes.scatter(x=times_with_asymmetric_fate_h, y=[0] * len(times_with_asymmetric_fate_h),
                 color=HISTOGRAM_BLUE, marker="o", label="Asymmetric fate")
    axes.scatter(x=times_with_unknown_fate_h, y=[0] * len(times_with_unknown_fate_h),
                 color=(0.3, 0.3, 0.3), marker="s", label="Unknown fate")

    all_times = times_with_symmetric_fate_h + times_with_asymmetric_fate_h + times_with_unknown_fate_h
    if len(all_times) > 0:
        axes.set_xlim(0, max(all_times) + 5)
    axes.set_ylim(0.1, -1)
    axes.set_yticks([])
    for side in ["top", "right", "left"]:
        axes.spines[side].set_visible(False)
    axes.set_xlabel("Time since last division (h)")
    axes.legend()
    figure.tight_layout()
