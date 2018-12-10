from typing import Dict, Any, Optional

import numpy
from matplotlib.figure import Figure
from numpy import ndarray

from autotrack.core import UserError, Name
from autotrack.core.links import ParticleLinks, LinkingTrack
from autotrack.gui import dialog
from autotrack.gui.gui_experiment import GuiExperiment
from autotrack.gui.threading import Task
from autotrack.gui.window import Window
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.lineage_division_count import get_division_count_in_lineage
from autotrack.linking_analysis.linking_markers import EndMarker

_LINEAGE_FOLLOW_TIME_H = 35


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        # Disabled while code is being updated for new linking structure
        "Graph/Clone size distribution-Clone size distribution...": lambda: _show_clone_size_distribution(window),
    }


def _show_clone_size_distribution(window: Window):
    experiment = window.get_experiment()
    links = experiment.links
    if not links.has_links():
        raise UserError("Failed to calculate clone size distribution",
                        "Cannot calculate clone size distribution. The linking data is missing.")
    try:
        experiment.image_resolution()
    except ValueError:
        raise UserError("Failed to calculate clone size distribution", "Cannot calculate clone size distribution. "
                                                                        "No time resolution is provided.")

    # Calculate the number of time points in the given follow time
    time_point_window = int(_LINEAGE_FOLLOW_TIME_H / experiment.image_resolution().time_point_interval_h)
    print("Time point window is", time_point_window)

    # Run the task on another thread, as calculating is quite slow
    window.get_scheduler().add_task(_CloneDistributionTask(window.get_gui_experiment(), links, time_point_window,
                                                            experiment.first_time_point_number(),
                                                            experiment.last_time_point_number()))


class _CloneDistributionTask(Task):

    _links: ParticleLinks
    _time_point_window: int
    _first_time_point_number: int
    _last_time_point_number: int
    _name: Name
    _gui_experiment: GuiExperiment

    def __init__(self, gui_experiment: GuiExperiment, links: ParticleLinks, time_point_window: int, first_time_point_number: int,
                 last_time_point_number: int):
        self._gui_experiment = gui_experiment
        self._links = links.copy()  # Copy so that we can safely access this on another thread
        self._time_point_window = time_point_window
        self._first_time_point_number = first_time_point_number
        self._last_time_point_number = last_time_point_number

    def compute(self) -> ndarray:
        return _get_clone_sizes_list(self._links, self._time_point_window, self._first_time_point_number,
                                      self._last_time_point_number)

    def on_finished(self, clone_sizes: ndarray):
        dialog.popup_figure(self._gui_experiment, lambda figure: _draw_clone_sizes(figure, clone_sizes))


def _get_clone_sizes_list(links: ParticleLinks, time_point_window: int, first_time_point_number: int, last_time_point_number: int) -> ndarray:
    clone_sizes = list()
    for view_start_time_point in range(first_time_point_number, last_time_point_number - time_point_window, 5):
        # Process a single time point window, from view_start_time_point to view_end_time_point
        view_end_time_point = view_start_time_point + time_point_window

        for track in links.find_all_tracks_in_time_point(view_start_time_point):
            # Process a single lineage in that time point
            division_count = get_division_count_in_lineage(track, links, view_end_time_point)
            if division_count is not None and division_count > 0:
                clone_sizes.append(division_count + 1)
    return numpy.array(clone_sizes, dtype=numpy.int32)


def _draw_clone_sizes(figure: Figure, clone_sizes: ndarray):
    max_clone_size = clone_sizes.max()

    axes = figure.gca()
    if len(clone_sizes) > 0:
        axes.hist(clone_sizes, range(1, max_clone_size + 2), color="blue")
    else:
        axes.text(0, 0, "No histogram: no lineages without dissappearing cells spanning at least"
                        f" {_LINEAGE_FOLLOW_TIME_H} h were found")
    axes.set_title("Clone size distribution")
    axes.set_ylabel("Relative frequency")
    axes.set_xlabel(f"Clone size after {_LINEAGE_FOLLOW_TIME_H} h")
    axes.set_xticks(range(2, max_clone_size + 1, 2))

