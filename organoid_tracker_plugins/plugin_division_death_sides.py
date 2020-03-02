"""Determines whether cell divisions and cell shedding events are happening on the left or the right side of the crypt-
villus axis, and whether there is a correlation between the sides."""
from typing import Dict, Any, Tuple

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.linking_analysis import linking_markers


class _CryptResult:
    """Bookkeeping of division and death counts."""
    left_divisions: int
    right_divisions: int
    left_deaths: int
    right_deaths: int

    def __init__(self):
        self.left_divisions, self.right_divisions = 0, 0
        self.left_deaths, self.right_deaths = 0, 0


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell cycle-Death and division events//Counts-Counts left and right of crypt-villus axis...": lambda: _view_division_and_death_sides(window),
    }


def _view_division_and_death_sides(window: Window):
    experiment = window.get_experiment()

    out: Dict[int, _CryptResult] = dict()
    _analyze_divisions(experiment, out)
    _analyze_deaths(experiment, out)

    if len(out) == 0:
        raise UserError("No divisions and deaths found", "No divisions and deaths found. Is the linking data"
                                                            " missing? Did you draw a crypt-villus axis?")

    message = "Note: left and right are defined in the direction of the crypt-villus axis.\n"
    for axis_id, result in out.items():
        message += f"\nAxis {axis_id}:\nCell divisions, left: {result.left_divisions}, right: {result.right_divisions}"
        message += f"\nCell deaths, left: {result.left_deaths}, right: {result.right_deaths}\n"
    dialog.popup_message("Results of death and division counting", message)


def _analyze_divisions(experiment: Experiment, out: Dict[int, _CryptResult]):
    """Counts how many divisions are happening in the right and left sides of the crypt."""
    splines = experiment.splines
    resolution = experiment.images.resolution()
    for mother_position in cell_division_finder.find_mothers(experiment.links):
        axis_position = splines.to_position_on_original_axis(experiment.links, mother_position)
        if axis_position is None:
            continue

        axis_out = out.get(axis_position.axis_id)
        if axis_out is None:
            axis_out = _CryptResult()
            out[axis_position.axis_id] = axis_out

        if axis_position.calculate_angle(mother_position, resolution) < 0:
            axis_out.left_divisions += 1
        else:
            axis_out.right_divisions += 1


def _analyze_deaths(experiment: Experiment, out: Dict[int, _CryptResult]):
    """Counts how many cell shedding events are happening in the right and left sides of the crypt."""
    splines = experiment.splines
    resolution = experiment.images.resolution()
    for dead_cell in linking_markers.find_death_and_shed_positions(experiment.links):
        axis_position = splines.to_position_on_original_axis(experiment.links, dead_cell)
        if axis_position is None:
            continue

        axis_out = out.get(axis_position.axis_id)
        if axis_out is None:
            axis_out = _CryptResult()
            out[axis_position.axis_id] = axis_out

        if axis_position.calculate_angle(dead_cell, resolution) < 0:
            axis_out.left_deaths += 1
        else:
            axis_out.right_deaths += 1
