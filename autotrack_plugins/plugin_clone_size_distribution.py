from typing import Dict, Any

import numpy
from matplotlib.figure import Figure
from numpy import ndarray

from ai_track.core import UserError, Name
from ai_track.core.links import Links
from ai_track.gui import dialog
from ai_track.gui.gui_experiment import GuiExperiment
from ai_track.gui.threading import Task
from ai_track.gui.window import Window
from ai_track.linking_analysis.lineage_division_count import get_division_count_in_lineage

_LINEAGE_FOLLOW_TIME_H = 40


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Lineages-Clone size distribution...": lambda: _show_clone_size_distribution(window),
    }


def _show_clone_size_distribution(window: Window):
    experiment = window.get_experiment()
    links = experiment.links
    if not links.has_links():
        raise UserError("Failed to calculate clone size distribution",
                        "Cannot calculate clone size distribution. The linking data is missing.")
    try:
        experiment.images.resolution()
    except ValueError:
        raise UserError("Failed to calculate clone size distribution", "Cannot calculate clone size distribution. "
                                                                        "No time resolution is provided.")

    # Calculate the number of time points in the given follow time
    time_point_window = int(_LINEAGE_FOLLOW_TIME_H / experiment.images.resolution().time_point_interval_h)
    print("Time point window is", time_point_window)

    # Run the task on another thread, as calculating is quite slow
    clone_sizes = _get_clone_sizes_list(links, time_point_window, experiment.first_time_point_number(),
                                                            experiment.last_time_point_number())
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_clone_sizes(figure, clone_sizes))


def _get_clone_sizes_list(links: Links, time_point_window: int, first_time_point_number: int, last_time_point_number: int) -> ndarray:
    clone_sizes = list()
    for view_start_time_point in range(first_time_point_number, last_time_point_number - time_point_window, 5):
        # Process a single time point window, from view_start_time_point to view_end_time_point
        view_end_time_point = view_start_time_point + time_point_window

        for track in links.find_all_tracks_in_time_point(view_start_time_point):
            # Process a single lineage in that time point
            division_count = get_division_count_in_lineage(track, links, view_end_time_point)
            if division_count is not None:
                clone_sizes.append(division_count + 1)
    return numpy.array(clone_sizes, dtype=numpy.int32)


def _draw_clone_sizes(figure: Figure, clone_sizes: ndarray):
    axes = figure.gca()

    max_clone_size = 1
    if len(clone_sizes) > 0:
        max_clone_size = clone_sizes.max()
        axes.hist(clone_sizes, range(1, max_clone_size + 2), color="blue")
    else:
        axes.text(0, 0, "No histogram: no lineages without dissappearing cells spanning at least"
                        f" {_LINEAGE_FOLLOW_TIME_H} h were found")
    axes.set_title("Clone size distribution")
    axes.set_ylabel("Relative frequency")
    axes.set_xlabel(f"Clone size after {_LINEAGE_FOLLOW_TIME_H} h")
    axes.set_xticks(range(2, max_clone_size + 1, 2))

