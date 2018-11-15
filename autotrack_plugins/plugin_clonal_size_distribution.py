from typing import Dict, Any

import numpy
from matplotlib.figure import Figure
from networkx import Graph
from numpy import ndarray

from autotrack.core import UserError
from autotrack.core.particles import Particle
from autotrack.gui import Window, dialog
from autotrack.linking import existing_connections
from autotrack.linking_analysis import cell_appearance_finder


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph/Clonal size distribution-Clonal size distribution...": lambda: _show_clonal_size_distribution(window),
    }


def _show_clonal_size_distribution(window: Window):
    experiment = window.get_experiment()
    graph = experiment.links.graph
    if graph is None:
        raise UserError("Failed to calculate clonal size distribution",
                        "Cannot calculate clonal size distribution. The linking data is missing.")
    clonal_sizes = _get_clonal_sizes_list(graph)
    dialog.popup_figure(experiment.name, lambda figure: _draw_clonal_sizes(figure, clonal_sizes))


def _get_clonal_sizes_list(graph: Graph) -> ndarray:
    clonal_sizes = list()
    for lineage_start in cell_appearance_finder.find_appeared_cells(graph):
        clonal_sizes.append(_get_division_count_in_lineage(lineage_start, graph) + 1)
    return numpy.array(clonal_sizes, dtype=numpy.int32)


def _get_division_count_in_lineage(particle: Particle, graph: Graph):
    """Gets how many divisions there are in the lineage starting at the given cell. If the cell does not divide, then
    this method will return 0."""
    clonal_size = 0
    while True:
        next_particles = existing_connections.find_future_particles(graph, particle)
        if len(next_particles) == 0:
            # Cell death/disappearance
            return clonal_size
        if len(next_particles) > 1:
            # Cell division, process daughters
            clonal_size += 1
            for next_particle in next_particles:
                clonal_size += _get_division_count_in_lineage(next_particle, graph)
            return clonal_size
        particle = next_particles.pop()


def _draw_clonal_sizes(figure: Figure, clonal_sizes: ndarray):
    axes = figure.gca()
    axes.hist(clonal_sizes, range(clonal_sizes.max() + 1), color="blue")
    axes.set_title("Clonal size distribution")
    axes.set_ylabel("Frequency")
    axes.set_xlabel("Clonal size")

