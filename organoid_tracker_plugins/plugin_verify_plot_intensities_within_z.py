import math
from collections import defaultdict
from typing import Dict, Any, List, Optional

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Verify intensities//WithinZ-Plot intensities within a z-layer (this time point)...":
            lambda: _plot_intensities_single_time_point(window),
        "Intensity//Record-Verify intensities//WithinZ-Plot intensities within a z-layer (all time points)...":
            lambda: _plot_intensities_all_time_points(window)
    }


def _draw_intensities_by_z(figure: Figure, time_point: Optional[TimePoint],
                           intensities_by_name_and_z: Dict[str, Dict[int, List[float]]]):
    """Creates a figure, one panel per intensity. The title will include the time point, or "all time points" if
    time_point is None."""
    intensity_keys = intensities_by_name_and_z.keys()
    axes: List[Axes] = list(numpy.array(figure.subplots(nrows=min(2, len(intensity_keys)),
                                                        ncols=math.ceil(len(intensity_keys) / 2),
                                                        sharex=True, sharey=True)).flatten())

    ax_index = 0
    for intensity_key, values_by_z in intensities_by_name_and_z.items():
        ax = axes[ax_index]
        ax.set_ylabel("Intensity/px (a.u.)")
        ax.set_xlabel("Z (px)")
        ax.set_title(intensity_key)

        color = SANDER_APPROVED_COLORS[ax_index % len(SANDER_APPROVED_COLORS)]

        z_values = list(values_by_z.keys())
        if len(z_values) > 0:
            intensities = list(values_by_z.values())
            violin = ax.violinplot(intensities, z_values, showmeans=False, showextrema=False, showmedians=True)
            for body in violin["bodies"]:
                body.set_color(color)
                body.set_alpha(1)
            violin["cmedians"].set_color("#2d3436")
        else:
            ax.text(0.5, 0.5, "No intensities found", transform=ax.transAxes, horizontalalignment="center",
                    verticalalignment="center")
        ax_index += 1

    if len(intensity_keys) > 1 and len(intensity_keys) % 2 != 0:
        # We will have an extra empty plot at the end, remove it
        axes[-1].axis("off")
    if time_point is None:
        figure.suptitle("Intensities (all time points)")
    else:
        figure.suptitle(f"Intensities (time point {time_point.time_point_number()})")
    figure.tight_layout()


def _plot_intensities_single_time_point(window: Window):
    intensities_by_name_and_z = dict()
    time_point = window.display_settings.time_point
    for experiment in window.get_active_experiments():
        for intensity_key in intensity_calculator.get_intensity_keys(experiment):
            if intensity_key not in intensities_by_name_and_z:
                intensities_by_name_and_z[intensity_key] = defaultdict(list)

            for position in experiment.positions.of_time_point(time_point):
                intensity = intensity_calculator.get_normalized_intensity(experiment, position,
                                                                          intensity_key=intensity_key, per_pixel=True)
                if intensity is None:
                    continue
                z = int(round(position.z))
                intensities_by_name_and_z[intensity_key][z].append(intensity)

    if len(intensities_by_name_and_z) == 0:
        raise UserError("No intensities found", f"No intensities found at time point"
                                                f" {time_point.time_point_number()}. Is the intensity data missing for this time point?")
    dialog.popup_figure(window, lambda figure: _draw_intensities_by_z(figure, time_point, intensities_by_name_and_z),
                        size_cm=(30, 25))


def _plot_intensities_all_time_points(window: Window):
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

    if len(intensities_by_name_and_z) == 0:
        raise UserError("No intensities found", f"No intensities found"
                                                "Is the intensity data missing?")
    dialog.popup_figure(window, lambda figure: _draw_intensities_by_z(figure, None, intensities_by_name_and_z),
                        size_cm=(30, 25))
