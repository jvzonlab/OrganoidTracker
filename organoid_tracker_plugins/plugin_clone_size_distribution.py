from typing import Dict, Any

import numpy
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core import UserError
from organoid_tracker.core.links import Links
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis.lineage_division_counter import get_division_count_in_lineage


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Misc-Clone size distribution...": lambda: _show_clone_size_distribution(window),
    }


def _show_clone_size_distribution(window: Window):
    experiment = window.get_experiment()
    links = experiment.links
    position_data = experiment.position_data
    if not links.has_links():
        raise UserError("Failed to calculate clone size distribution",
                        "Cannot calculate clone size distribution. The linking data is missing.")
    try:
        experiment.images.resolution()
    except ValueError:
        raise UserError("Failed to calculate clone size distribution", "Cannot calculate clone size distribution. "
                                                                        "No time resolution is provided.")

    lineage_follow_time_h = dialog.prompt_float("Time window", "The clone size is the number of progeny a single cell"
                                                               " generated after a certain time window.\nHow long"
                                                               " should this time window be, in hours?", default=40,
                                                minimum=1)
    if lineage_follow_time_h is None:
        return

    # Calculate the number of time points in the given follow time
    time_point_window = int(lineage_follow_time_h / experiment.images.resolution().time_point_interval_h)

    clone_sizes = _get_clone_sizes_list(links, position_data, time_point_window, experiment.first_time_point_number(),
                                                            experiment.last_time_point_number())
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_clone_sizes(figure, lineage_follow_time_h, clone_sizes))


def _get_clone_sizes_list(links: Links, position_data: PositionData, time_point_window: int,
                          first_time_point_number: int, last_time_point_number: int) -> ndarray:
    clone_sizes = list()
    for view_start_time_point in range(first_time_point_number, last_time_point_number - time_point_window, 5):
        # Process a single time point window, from view_start_time_point to view_end_time_point
        view_end_time_point = view_start_time_point + time_point_window

        for track in links.find_all_tracks_in_time_point(view_start_time_point):
            # Process a single lineage in that time point
            division_count = get_division_count_in_lineage(track, position_data, view_end_time_point)
            if division_count is not None:
                clone_sizes.append(division_count + 1)
    return numpy.array(clone_sizes, dtype=numpy.int32)


def _draw_clone_sizes(figure: Figure, lineage_follow_time_h: float, clone_sizes: ndarray):
    axes = figure.gca()

    max_clone_size = 1
    if len(clone_sizes) > 0:
        max_clone_size = clone_sizes.max()
        axes.hist(clone_sizes, range(1, max_clone_size + 2), color="blue")
    else:
        axes.text(0, 0, "No histogram: no lineages without dissappearing cells spanning at least"
                        f" {lineage_follow_time_h} h were found")
    axes.set_title("Clone size distribution")
    axes.set_ylabel("Relative frequency")
    axes.set_xlabel(f"Clone size after {lineage_follow_time_h} h")
    axes.set_xticks(range(2, max_clone_size + 1, 2))

