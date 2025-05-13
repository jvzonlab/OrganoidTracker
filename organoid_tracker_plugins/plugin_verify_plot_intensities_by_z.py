import math
from collections import defaultdict
from typing import Dict, Any, List

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Verify intensities//LineGraph-Plot intensities by z...": lambda: _plot_intensities_by_z(window)
    }


def _draw_intensities_by_z(figure: Figure, intensities_by_name_and_z: Dict[str, Dict[int, List[float]]]):
    ax: Axes = figure.gca()

    i = 0
    for intensity_key, values_by_z in intensities_by_name_and_z.items():
        z_values = numpy.arange(min(values_by_z.keys()), max(values_by_z.keys()) + 1)
        intensity_means = numpy.full_like(z_values, fill_value=numpy.nan, dtype=numpy.float64)
        intensity_stds = numpy.full_like(z_values, fill_value=numpy.nan, dtype=numpy.float64)
        for z, values in values_by_z.items():
            z_index = z - z_values[0]
            intensity_means[z_index] = numpy.mean(values)
            intensity_stds[z_index] = numpy.std(values, ddof=1)

        color = SANDER_APPROVED_COLORS[i % len(SANDER_APPROVED_COLORS)]
        ax.plot(z_values, intensity_means, label=intensity_key, color=color, linewidth=3)
        ax.fill_between(z_values, intensity_means - intensity_stds, intensity_means + intensity_stds, color=color, alpha=0.4)
        i += 1

    ax.set_ylabel("Intensity/px (a.u.)")
    ax.set_xlabel("Z (px)")
    if len(intensities_by_name_and_z) > 1:
        ax.legend()


def _plot_intensities_by_z(window: Window):
    intensities_by_name_and_z = dict()
    for experiment in window.get_active_experiments():
        for intensity_key in intensity_calculator.get_intensity_keys(experiment):
            if intensity_key not in intensities_by_name_and_z:
                intensities_by_name_and_z[intensity_key] = defaultdict(list)

            for position, _ in experiment.position_data.find_all_positions_with_data(intensity_key):
                intensity = intensity_calculator.get_normalized_intensity(experiment, position,
                                                                          intensity_key=intensity_key, per_pixel=True)
                if intensity is None:
                    continue
                z = int(round(position.z))
                intensities_by_name_and_z[intensity_key][z].append(intensity)

    dialog.popup_figure(window, lambda figure: _draw_intensities_by_z(figure, intensities_by_name_and_z))


