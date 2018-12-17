from typing import Dict, Any, List

import numpy
from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking_analysis import linking_markers


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell deaths-Cell deaths//Graph-Locations on crypt axis...": lambda: _view_cell_death_locations(window),
    }


def _view_cell_death_locations(window: Window):
    experiment = window.get_experiment()
    data_axes = experiment.data_axes
    resolution = experiment.image_resolution()

    if not data_axes.has_axes():
        raise UserError("Dead cells", "No crypt axes where found. Cannnot determine positions of dead cells.")

    dead_cells = linking_markers.find_dead_particles(experiment.links)
    crypt_positions = dict()
    for dead_cell in dead_cells:
        crypt_position = data_axes.to_position_on_axis(dead_cell)
        if crypt_position is not None:
            if crypt_position.axis_id not in crypt_positions:
                crypt_positions[crypt_position.axis_id] = []
            crypt_positions[crypt_position.axis_id].append(crypt_position.pos * resolution.pixel_size_x_um)

    if len(crypt_positions) == 0:
        raise UserError("Dead cells", "No cell deaths were found. Did you mark any lineage ends as actual cell deaths?")

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cell_deaths(figure, crypt_positions))


def _draw_cell_deaths(figure: Figure, crypt_positions: Dict[int, List[float]]):
    ticks = numpy.arange(0, int(_get_highest_crypt_position(crypt_positions)) + 6, 5)
    colors = ((0, 1, 0, 0.5), (0, 0, 0, 0.5), (1, 0, 0, 0.5))

    axes = figure.gca()
    axes.set_title("Positions of cell deaths on crypt axes")
    axes.set_xticks(ticks)
    for axis_id, single_crypt_positions in crypt_positions.items():
        axes.hist(single_crypt_positions, bins=ticks, label=f"Axis {axis_id}", color=colors[axis_id % len(colors)])
    axes.set_xlabel("Position on crypt axis (μm)")
    axes.set_ylabel("Amount of cell deaths")
    if len(crypt_positions) > 1:
        axes.legend()


def _get_highest_crypt_position(crypt_positions: Dict[int, List[float]]) -> float:
    max_positions = [max(positions) for positions in crypt_positions.values()]
    return max(max_positions)

