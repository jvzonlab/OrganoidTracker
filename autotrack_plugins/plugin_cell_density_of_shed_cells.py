
from typing import Dict, Any, List

import math
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ai_track.core import UserError
from ai_track.core.experiment import Experiment
from ai_track.gui import dialog
from ai_track.gui.window import Window
from ai_track.linking_analysis import linking_markers, cell_compartment_finder
from ai_track.position_analysis import cell_density_calculator
from ai_track.linking_analysis.cell_compartment_finder import CellCompartment
from ai_track.util.mpl_helper import BAR_COLOR_1, BAR_COLOR_2


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell cycle-Cell density//Average cell density around shedding...": lambda: _show_cell_densities(window),
    }


def _show_cell_densities(window: Window):
    experiment = window.get_experiment()
    shed_densities = _get_cell_densities_of_shed_cells_in_nondividing_compartment_mm1(experiment)
    if len(shed_densities) == 0:
        raise UserError("Densities of shed cells", "No cell shedding was found in the non-dividing compartment. Note"
                                                   f" that the last {experiment.division_lookahead_time_points} time"
                                                   f" points of the experiment are ignored, as it cannot be determined"
                                                   f" there whether cells are dividing or not.")
    all_densities = _get_all_cell_densities_in_nondividing_compartment_mm1(experiment)
    dialog.popup_figure(window.get_gui_experiment(),
                        lambda figure: _plot_cell_densities(figure, all_densities, shed_densities), size_cm=(7,6))


def _plot_cell_densities(figure: Figure, all_densities: List[float], shed_densities: List[float]):
    max_density = math.ceil(max(all_densities))
    axes = figure.gca()

    axes.hist(shed_densities, bins=range(0, max_density, 10), label="Just the shed cells", alpha=0.7,
              color=BAR_COLOR_1)
    axes.tick_params('y', colors=BAR_COLOR_1)
    axes.set_ylabel("Amount of shed cells")

    axes_twin: Axes = axes.twinx()
    axes_twin.hist(all_densities, bins=range(0, max_density, 2), label="All densities in the villus", alpha=0.7,
                   color=BAR_COLOR_2)
    axes_twin.set_ylabel("Total amount of cell detections in the villus")
    axes_twin.tick_params('y', colors=BAR_COLOR_2)
    axes.set_xlabel("Cell density (mm$^{-1}$)")


def _get_all_cell_densities_in_nondividing_compartment_mm1(experiment: Experiment):
    """Gets all densities just before cell shedding in the villus."""
    densities = list()
    resolution = experiment.images.resolution()

    for time_point in experiment.time_points():
        positions_of_time_point = experiment.positions.of_time_point(time_point)
        for position in positions_of_time_point:
            if cell_compartment_finder.find_compartment(experiment, position) != CellCompartment.NON_DIVIDING:
                continue
            densities.append(cell_density_calculator.get_density_mm1(positions_of_time_point, position, resolution))

    return densities


def _get_cell_densities_of_shed_cells_in_nondividing_compartment_mm1(experiment: Experiment) -> List[float]:
    """Gets all densities just before cell shedding in the villus."""
    densities = list()

    links = experiment.links
    resolution = experiment.images.resolution()
    for shed_position in linking_markers.find_shed_positions(links):
        # Get position two time points back (so before shedding)
        past_positions = links.find_pasts(shed_position)
        if len(past_positions) != 1:
            continue
        past_positions = links.find_pasts(past_positions.pop())
        if len(past_positions) != 1:
            continue
        past_position = past_positions.pop()

        if cell_compartment_finder.find_compartment(experiment, past_position) != CellCompartment.NON_DIVIDING:
            continue

        positions_in_time_point = experiment.positions.of_time_point(past_position.time_point())
        densities.append(cell_density_calculator.get_density_mm1(positions_in_time_point, past_position, resolution))

    return densities
