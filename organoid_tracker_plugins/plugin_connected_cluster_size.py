"""Counts the size of each cluster. Clusters are formed using connnections (see the Connections class)."""
from typing import Dict, Any, List

import numpy
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.connecting import cluster_finder
from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Clusters-Connected cluster size...": lambda: _show_connected_cluster_size(window)
    }

def _show_connected_cluster_size(window: Window):
    if not dialog.popup_message_cancellable("Connected cluster size", "You can connect two different positions in a time"
              " point by inserting connections. These connections are used to create clusters of positions. This"
              " function counts how many clusters there are of each size.\n\n"
              "Note: this dialog counts accross all time points."):
        return
    recorded_cluster_sizes = list()
    highest_recorded_size = 0
    experiment = window.get_experiment()
    positions = experiment.positions
    connections = experiment.connections
    for time_point in experiment.time_points():
        for cluster in cluster_finder.find_clusters(positions, connections, time_point):
            cluster_size = len(cluster.positions)
            recorded_cluster_sizes.append(cluster_size)
            if cluster_size > highest_recorded_size:
                highest_recorded_size = cluster_size

    if highest_recorded_size == 0:
        raise UserError("Connected cluster size", "Didn't find any clusters. Are there no positions loaded?")

    bins = numpy.arange(0, highest_recorded_size + 1)
    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _show_graph(figure, bins, recorded_cluster_sizes))

def _show_graph(figure: Figure, bins: ndarray, recorded_cluster_sizes: List[int]):
    axes = figure.gca()
    values, _, _ = axes.hist(recorded_cluster_sizes, bins=bins)
    axes.set_xlabel("Cluster size")
    axes.set_ylabel("Count")
    for cluster_size, amount in zip(bins, values):
        print(cluster_size, amount)
