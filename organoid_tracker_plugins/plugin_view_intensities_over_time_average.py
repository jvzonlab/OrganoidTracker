from functools import partial
from typing import NamedTuple, List, Dict, Any

import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS


def get_menu_items(window: Window) -> Dict[str, Any]:
    # Dynamic menu entries depending on the available intensities

    # Collect intensities
    intensity_keys = set()
    for experiment in window.get_active_experiments():
        intensity_keys |= set(intensity_calculator.get_intensity_keys(experiment))

    if len(intensity_keys) == 0:
        # Just show an error message if clicked
        return {
            "Intensity//View-View average intensities over time...":
                lambda: dialog.popup_error("No intensities recorded",
                                           "No intensities were recorded. Please do so first")
        }
    elif len(intensity_keys) == 1:
        # Just use the only intensity
        return {
            "Intensity//View-View average intensities over time...": lambda: _view_intensities(window, intensity_keys.pop())
        }
    else:
        # Separate menu option for each
        return_value = dict()
        for intensity_key in intensity_keys:
            return_value["Intensity//View-View average intensities over time//" + intensity_key] \
                = partial(_view_intensities, window, intensity_key)
        return return_value


class _IntensitiesOfExperiment(NamedTuple):
    mean_values: ndarray
    std_values: ndarray
    times_h: ndarray


def _view_intensities(window: Window, intensity_key: str):
    all_intensities = []
    for experiment in window.get_active_experiments():
        all_intensities.append(_collect_intensities(experiment, intensity_key))

    dialog.popup_figure(window, lambda figure: _plot(figure, all_intensities))


def _plot(figure: Figure, all_intensities: List[_IntensitiesOfExperiment]):
    ax: Axes = figure.gca()
    max_value = 0
    for i, intensities in enumerate(all_intensities):
        color = SANDER_APPROVED_COLORS[i % len(SANDER_APPROVED_COLORS)]
        ax.plot(intensities.times_h, intensities.mean_values, color=color, linewidth=3)
        ax.fill_between(intensities.times_h, intensities.mean_values - intensities.std_values,
                        intensities.mean_values + intensities.std_values, color=color, alpha=0.3)
        max_value = max(max_value, (intensities.mean_values + intensities.std_values).max())
    ax.set_ylim(0, max_value * 1.1)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Intensity (±st.dev.)")


def _collect_intensities(experiment: Experiment, intensity_key: str):
    timings = experiment.images.timings()

    times_h = list()
    mean_values = list()
    std_values = list()
    for time_point in experiment.positions.time_points():
        intensities = list()
        for position in experiment.positions.of_time_point(time_point):
            intensity = intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=intensity_key)
            if intensity is not None:
                intensities.append(intensity)

        if len(intensities) > 0:
            times_h.append(timings.get_time_h_since_start(time_point))
            mean_values.append(numpy.mean(intensities))
            std_values.append(numpy.std(intensities, ddof=1) if len(intensities) > 1 else 0)

    return _IntensitiesOfExperiment(mean_values=numpy.array(mean_values),
                                    std_values=numpy.array(std_values),
                                    times_h=numpy.array(times_h))
