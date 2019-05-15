from typing import Dict, Any, List, Tuple

import numpy
from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking import cell_division_finder
from autotrack.linking_analysis import linking_markers, cell_density_calculator
from autotrack.util.moving_average import MovingAverage


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell cycle-Death and division events//Graph-Locations on crypt axis...": lambda: _view_cell_death_locations(window),
    }


def _view_cell_death_locations(window: Window):
    experiment = window.get_experiment()
    data_axes = experiment.splines
    resolution = experiment.images.resolution()

    if not data_axes.has_splines():
        raise UserError("Dead cells", "No crypt axes where found. Cannot determine positions of dead cells.")

    death_crypt_positions = dict()
    for dead_cell in linking_markers.find_death_and_shed_positions(experiment.links):
        crypt_position = data_axes.to_position_on_original_axis(experiment.links, dead_cell)
        if crypt_position is not None:
            if crypt_position.axis_id not in death_crypt_positions:
                death_crypt_positions[crypt_position.axis_id] = []
            death_crypt_positions[crypt_position.axis_id].append(crypt_position.pos * resolution.pixel_size_x_um)

    mother_crypt_positions = dict()
    for mother_cell in cell_division_finder.find_mothers(experiment.links):
        crypt_position = data_axes.to_position_on_original_axis(experiment.links, mother_cell)
        if crypt_position is not None:
            if crypt_position.axis_id not in mother_crypt_positions:
                mother_crypt_positions[crypt_position.axis_id] = []
            mother_crypt_positions[crypt_position.axis_id].append(crypt_position.pos * resolution.pixel_size_x_um)

    cell_densities = _get_cell_densities(experiment)

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cell_events(figure, mother_crypt_positions,
            death_crypt_positions, cell_densities))


def _draw_cell_events(figure: Figure, mother_crypt_positions: Dict[int, List[float]],
                      death_crypt_positions: Dict[int, List[float]],
                      cell_densities: Dict[int, Tuple[List[float], List[float]]]):
    highest_pos = max(_get_highest_crypt_position(death_crypt_positions),
                      _get_highest_crypt_position(mother_crypt_positions))
    # Get list of all used axis numbers, without duplicates
    axis_ids = list(dict.fromkeys(death_crypt_positions.keys() | mother_crypt_positions.keys()))
    bin_size = 5
    bins = numpy.arange(0, int(highest_pos) + bin_size + 1, bin_size)
    ticks = bins if highest_pos < 90 else numpy.arange(0, int(highest_pos) + 2 * bin_size + 1, 2 * bin_size)

    axes = figure.subplots(len(axis_ids), sharex=True) if len(axis_ids) > 1 else [figure.gca()]
    death_color = (1, 0, 0, 0.7)
    division_color = (0, 0, 1, 0.5)
    for i in range(len(axis_ids)):
        axis = axes[i]
        axis_id = axis_ids[i]
        axis.set_title(f"Crypt-villus axis {axis_id}")
        axis.set_xticks(ticks)
        axis.set_xlim(0, highest_pos)
        if axis_id in death_crypt_positions:
            axis.hist(death_crypt_positions[axis_id], bins=bins, label=f"Deaths", color=death_color)
        if axis_id in mother_crypt_positions:
            axis.hist(mother_crypt_positions[axis_id], bins=bins, label=f"Divisions", color=division_color)
        if axis_id in cell_densities:
            scaled_axis = axis.twinx()
            scaled_axis.set_ylabel("Cell density (μm$^{-1}$)")
            scaled_axis.tick_params(axis='y', labelcolor="lime")
            moving_average = MovingAverage(cell_densities[axis_id][0], cell_densities[axis_id][1], window_width=bin_size)
            moving_average.plot(scaled_axis, color="lime")
        axis.set_ylabel("Amount")
        if i == 0:
            axis.legend()  # First panel, show legend
        if i == len(axis_ids) - 1:  # Last panel, show x label
            axis.set_xlabel("Position on crypt axis (μm)")


def _get_highest_crypt_position(crypt_positions: Dict[int, List[float]]) -> float:
    max_positions = [max(positions) for positions in crypt_positions.values()]
    return max(max_positions)


def _get_cell_densities(experiment: Experiment) -> Dict[int, Tuple[List[float], List[float]]]:
    """Gets all cell densities. For each axis, two lists are stored: crypt-villus positions with corresponding cell
    densities."""
    links = experiment.links
    resolution = experiment.images.resolution()

    result = dict()

    for time_point in experiment.time_points():
        for position in experiment.positions.of_time_point(time_point):
            axis_position = experiment.splines.to_position_on_original_axis(links, position)
            if axis_position is None:
                continue

            density = cell_density_calculator.get_density(
                experiment.positions.of_time_point(time_point), position, resolution)

            result_tuple = result.get(axis_position.axis_id)
            if result_tuple is None:
                result_tuple = (list(), list())
                result[axis_position.axis_id] = result_tuple

            result_tuple[0].append(axis_position.pos * resolution.pixel_size_x_um)
            result_tuple[1].append(density)

    return result
