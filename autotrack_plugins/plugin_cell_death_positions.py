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
    crypt_positions = list()
    for dead_cell in dead_cells:
        crypt_position = data_axes.to_position_on_axis(dead_cell)
        if crypt_position is not None:
            crypt_positions.append(crypt_position.pos * resolution.pixel_size_x_um)

    if len(crypt_positions) == 0:
        raise UserError("Dead cells", "No cell deaths were found. Did you mark any lineage ends as actual cell deaths?")

    dialog.popup_figure(window.get_gui_experiment(), lambda figure: _draw_cell_deaths(figure, crypt_positions))


def _draw_cell_deaths(figure: Figure, crypt_positions: List[float]):
    axes = figure.gca()
    axes.set_title("Positions of cell deaths on crypt axes")
    axes.hist(crypt_positions)
    axes.set_xlabel("Position on crypt axis (μm)")
    axes.set_ylabel("Amount of cell deaths")
