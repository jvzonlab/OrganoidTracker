from typing import Dict, Any

from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer.lineage_tree_visualizer import LineageTreeVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Lineages-Interactive lineage tree...": lambda: _show_lineage_tree(window)
    }


def _show_lineage_tree(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")

    dialog.popup_visualizer(window, LineageTreeVisualizer)
