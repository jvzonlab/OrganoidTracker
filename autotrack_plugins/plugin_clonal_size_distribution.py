from typing import Dict, Any, Optional

import numpy
from matplotlib.figure import Figure
from networkx import Graph
from numpy import ndarray

from autotrack.core import UserError, Name
from autotrack.core.links import ParticleLinks
from autotrack.core.particles import Particle
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.gui.threading import Task
from autotrack.linking_analysis import linking_markers
from autotrack.linking_analysis.linking_markers import EndMarker


_LINEAGE_FOLLOW_TIME_H = 25


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph/Clonal size distribution-Clonal size distribution...": lambda: _show_clonal_size_distribution(window),
    }


def _show_clonal_size_distribution(window: Window):
    experiment = window.get_experiment()
    links = experiment.links
    if not links.has_links():
        raise UserError("Failed to calculate clonal size distribution",
                        "Cannot calculate clonal size distribution. The linking data is missing.")
    try:
        experiment.image_resolution()
    except ValueError:
        raise UserError("Failed to calculate clonal size distribution", "Cannot calculate clonal size distribution. "
                                                                        "No time resolution is provided.")

    # Calculate the number of time points in the given follow time
    time_point_window = int(_LINEAGE_FOLLOW_TIME_H * 60 / experiment.image_resolution().time_point_interval_m)
    print("Time point window is", time_point_window)

    # Run the task on another thread, as calculating is quite slow
    window.get_scheduler().add_task(_ClonalDistributionTask(experiment.name, links, time_point_window,
                                                            experiment.first_time_point_number(),
                                                            experiment.last_time_point_number()))


class _ClonalDistributionTask(Task):

    _links: ParticleLinks
    _time_point_window: int
    _first_time_point_number: int
    _last_time_point_number: int
    _name: Name

    def __init__(self, name: Name, links: ParticleLinks, time_point_window: int, first_time_point_number: int,
                 last_time_point_number: int):
        self._name = name
        self._links = links.copy()  # Copy so that we can safely access this on another thread
        self._time_point_window = time_point_window
        self._first_time_point_number = first_time_point_number
        self._last_time_point_number = last_time_point_number

    def compute(self) -> ndarray:
        return _get_clonal_sizes_list(self._links, self._time_point_window, self._first_time_point_number,
                                      self._last_time_point_number)

    def on_finished(self, clonal_sizes: ndarray):
        dialog.popup_figure(self._name, lambda figure: _draw_clonal_sizes(figure, clonal_sizes))


def _get_clonal_sizes_list(links: ParticleLinks, time_point_window: int, first_time_point_number: int, last_time_point_number: int) -> ndarray:
    clonal_sizes = list()
    for view_start_time_point in range(first_time_point_number, last_time_point_number - time_point_window, 5):
        print("Calculating clonal sizes at time point", view_start_time_point)
        view_end_time_point = view_start_time_point + time_point_window
        sublinks = links.limit_to_time_points(view_start_time_point, view_end_time_point)
        for lineage_start in sublinks.find_appeared_cells():
            cell_divisions_count = _get_division_count_in_lineage(lineage_start, sublinks, view_end_time_point)
            if cell_divisions_count is not None:
                clonal_sizes.append(cell_divisions_count + 1)  # +1 to convert cell divisions count to clonal size
    return numpy.array(clonal_sizes, dtype=numpy.int32)


def _get_division_count_in_lineage(particle: Particle, links: ParticleLinks, last_time_point_number: int) -> Optional[int]:
    """Gets how many divisions there are in the lineage starting at the given cell. If the cell does not divide, then
    this method will return 0."""
    division_count = 0
    while True:
        next_particles = links.find_futures(particle)
        if len(next_particles) == 0:
            # Cell death/disappearance
            end_marker = linking_markers.get_track_end_marker(links, particle)
            if particle.time_point_number() == last_time_point_number or end_marker == EndMarker.DEAD:
                return division_count
            return None  # We cannot completely analyze this lineage, as a cell went out the field of view
        if len(next_particles) > 1:
            # Cell division, process daughters
            division_count += 1
            for next_particle in next_particles:
                next_division_count = _get_division_count_in_lineage(next_particle, links, last_time_point_number)
                if next_division_count is None:
                    return None  # Cannot determine number of divisions in this linage
                division_count += next_division_count
            return division_count
        particle = next_particles.pop()


def _draw_clonal_sizes(figure: Figure, clonal_sizes: ndarray):
    axes = figure.gca()
    if len(clonal_sizes) > 0:
        axes.hist(clonal_sizes, range(clonal_sizes.max() + 2), color="blue")
    else:
        axes.text(0, 0, "No histogram: no lineages without dissappearing cells spanning at least"
                        f" {_LINEAGE_FOLLOW_TIME_H} h were found")
    axes.set_title("Clonal size distribution")
    axes.set_ylabel("Relative frequency")
    axes.set_xlabel(f"Clonal size after {_LINEAGE_FOLLOW_TIME_H} h")

