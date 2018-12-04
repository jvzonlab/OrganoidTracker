from typing import Dict, Any

from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking_analysis import lineage_drawing
from autotrack.linking_analysis.lineage_drawing import LineageDrawing


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph/Lineages-Show lineage tree...": lambda: _show_lineage_tree(window)
    }


def _show_lineage_tree(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")

    experiment.links.debug_sanity_check()
    dialog.popup_figure(experiment.name, lambda fig: _plot_lineages(fig, experiment))


def _plot_lineages(fig: Figure, experiment: Experiment):
    axes = fig.gca()

    width = LineageDrawing(experiment.links).draw_lineages(axes, show_cell_id=False)

    axes.set_ylabel("Time (time points)")
    axes.set_ylim([experiment.last_time_point_number(), experiment.first_time_point_number() - 1])
    axes.set_xlim([-0.1, width + 0.1])
