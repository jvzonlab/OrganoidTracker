from typing import Dict, Any

from autotrack.gui.window import Window
from autotrack.linking import cell_division_finder
from autotrack.linking_analysis import linking_markers


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Lineages-Shedding events following divisions...": lambda: _show_sheddings_following_divisions(window)
    }


def _show_sheddings_following_divisions(window: Window):
    experiment = window.get_experiment()
    print("---")
    for shed_position in linking_markers.find_shed_positions(experiment.links):
        division = cell_division_finder.get_previous_division(experiment.links, shed_position)
        if division is not None:
            age = shed_position.time_point_number() - division.mother.time_point_number()
            print("Found shed cell", shed_position, "that divided at", division.mother, "which is", age, "time points earlier")


