from typing import Dict, Any, Optional

from matplotlib.figure import Figure
from networkx import Graph

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle
from autotrack.gui import Window, dialog
from autotrack.linking import cell_cycle, mother_finder

# Were no divisions found because a cell really didn't divide, or did the experiment simply end before the cell divided?
# If the experiment continues for at least this many time points, then we can safely assume that the cell did not
# divide.
_DIVISION_LOOKAHEAD = 100


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph/Cell cycle-Chance of division...": lambda: _show_chance_of_division(window.get_experiment())
    }


def _show_chance_of_division(experiment: Experiment):
    links = experiment.links.get_baseline_else_scratch()
    if links is None:
        raise UserError("No linking data found", "For this graph on cell divisions, it is required to have the cell"
                                                 " links loaded.")

    max_time_point_number = experiment.last_time_point_number()

    dialog.popup_figure(experiment.name, lambda figure: _draw_histogram(figure, links, max_time_point_number))


def _draw_histogram(figure: Figure, links: Graph, max_time_point_number: int):
    dividing_cells, nondividing_cells, unknown_fate_cells = _classify_cell_divisions(links, max_time_point_number)
    colors = ["orange", "blue", "lightgray"]
    labels = ["Dividing", "Non-dividing", "Unknown"]

    axes = figure.gca()
    axes.hist([dividing_cells, nondividing_cells, unknown_fate_cells], bins=10, histtype="barstacked", color=colors,
              label=labels)
    axes.set_xlabel("Duration of previous cell cycle (time points)")
    axes.set_ylabel("Occurrences")
    axes.set_title("Fate of cells dividing")
    axes.legend()


def _classify_cell_divisions(links: Graph, max_time_point_number: int):
    """Classifies each cell division in the data: is it the last cell division in a lineage, or will there be more after
    this? Returns three lists, each containing numbers representing the time duration of the previous generation."""
    dividing_cells = list()
    nondividing_cells = list()
    unknown_fate_cells = list()

    for cell_division in mother_finder.find_families(links):
        previous_cell_cycle_length = cell_cycle.get_age(links, cell_division.mother)
        if previous_cell_cycle_length is None:
            continue  # Cannot plot without knowing the length of the previous cell cycle
        for daughter in cell_division.daughters:
            will_divide = _will_divide(links, daughter, max_time_point_number)
            if will_divide is None:
                unknown_fate_cells.append(previous_cell_cycle_length)
            elif will_divide:  # so will_divide == True
                dividing_cells.append(previous_cell_cycle_length)
            else:  # will_divide == False
                nondividing_cells.append(previous_cell_cycle_length)
    return dividing_cells, nondividing_cells, unknown_fate_cells


def _will_divide(graph: Graph, particle: Particle, max_time_point_number: int) -> Optional[bool]:
    """Checks if a cell will undergo a division later in the experiment. Returns None if not sure, because we are near
    the end of the experiment. max_time_point_number is the number of the last time point in the experiment."""
    next_division = cell_cycle.get_next_division(graph, particle)
    if next_division is None:
        # Did the experiment end, or did the cell really stop dividing?
        if particle.time_point_number() + _DIVISION_LOOKAHEAD > max_time_point_number:
            return None  # We are not sure
        return False  # Cell with not divide
    return True  # Cell will divide in the future
