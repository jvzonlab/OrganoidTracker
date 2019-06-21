from typing import Dict, Any

import numpy
from numpy import ndarray

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.window import Window
from autotrack.linking_analysis import cell_compartment_finder
from autotrack.position_analysis import cell_density_calculator
from autotrack.linking_analysis.cell_compartment_finder import CellCompartment


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Graph//Cell cycle-Cell density//Average cell density in compartments...": lambda: _show_cell_density(window),
    }


def _show_cell_density(window: Window):
    densities_mm1 = _get_cell_densities(window.get_experiment())
    if len(densities_mm1) == 0:
        raise UserError("Cell density", "Found no cell positions - cannot report anything.")
    message = ""
    for compartment, densities_in_compartment in densities_mm1.items():
        message += f"\n{compartment.name.lower()}: Average ± standard deviation is {densities_in_compartment.mean():.1f}" \
            f" ± {densities_in_compartment.std(ddof=1):.1f} mm⁻¹"

    dialog.popup_message("Cell densities", "Cell densities: \n" + message)


def _get_cell_densities(experiment: Experiment) -> Dict[CellCompartment, ndarray]:
    densities_mm1 = dict()
    resolution = experiment.images.resolution()

    # Get all densities for all compartments
    for time_point in experiment.time_points():
        positions = experiment.positions.of_time_point(time_point)
        for position in positions:
            compartment = cell_compartment_finder.find_compartment(experiment, position)
            density = cell_density_calculator.get_density_mm1(positions, around=position, resolution=resolution)
            if compartment in densities_mm1:
                densities_mm1[compartment].append(density)
            else:
                densities_mm1[compartment] = [density]

    # Transform lists to numpy arrays and return
    densities_numpy_mm1 = dict()
    for compartment, values in densities_mm1.items():
        densities_numpy_mm1[compartment] = numpy.array(values, dtype=numpy.float64)
    return densities_numpy_mm1
