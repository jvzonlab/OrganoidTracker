from typing import Dict, Any

from ai_track.gui.window import Window
from ai_track.linking import cell_division_finder
from ai_track.linking_analysis import linking_markers, cell_fate_finder
from ai_track.linking_analysis.cell_fate_finder import CellFateType


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Cell cycle-Cell deaths//Graph-Fate of sisters of shed cells...": lambda: _show_sisters_of_shed_cells(window)
    }

def _show_sisters_of_shed_cells(window: Window):
    print("----")
    experiment = window.get_experiment()

    links = experiment.links
    for shed_cell in linking_markers.find_shed_positions(links):
        shedding_track = links.get_track(shed_cell)
        division = cell_division_finder.get_previous_division(links, shed_cell)
        if division is None:
            continue
        for daughter in division.daughters:
            if links.get_track(daughter) == shedding_track:
                continue
            # We now know that we have the sister of the shed cell

            print(cell_fate_finder.get_fate(experiment, shedding_track.find_first_position()), "vs",
                  cell_fate_finder.get_fate(experiment, daughter))

