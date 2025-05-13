import math
from collections import defaultdict
from typing import Dict, Any, List, Iterable, Tuple

import matplotlib
import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from organoid_tracker.core import Color
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.moving_average import MovingAverage, LinesAverage
from organoid_tracker.util.mpl_helper import SANDER_APPROVED_COLORS


_AVERAGING_TIME_H = 4


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Verify intensities//LineGraph-Plot intensities by cell cycle...": lambda: _plot_intensities_by_cell_cycle(
            window)
    }


class _IntensitiesOverTime:
    """The index in cell_cycle_percentages matches the data point with the same index in intensities."""
    lines: List[Tuple[List[float], List[float]]]

    def __init__(self):
        self.lines = list()

    def add_line(self, time_since_division_h: List[float], intensities: List[float]):
        self.lines.append((time_since_division_h, intensities))


def _draw_intensities_by_cell_cycle(figure: Figure, intensities_by_name: Dict[str, _IntensitiesOverTime]):
    print(intensities_by_name)
    ax: Axes = figure.gca()

    i = 0
    for intensity_key, values_by_t in intensities_by_name.items():
        color = SANDER_APPROVED_COLORS[i % len(SANDER_APPROVED_COLORS)]
        average = LinesAverage(*values_by_t.lines, x_step_size=0.05)
        average.plot(ax, color=Color.from_rgb_floats(*matplotlib.colors.to_rgb(color)), label=intensity_key)
        i += 1

    if len(intensities_by_name) > 1:
        ax.legend()
    ax.set_ylabel("Intensity/px (a.u.)")
    ax.set_xlabel("Time (h)")


def _find_tracks_connected_to_division(experiment: Experiment) -> Iterable[LinkingTrack]:
    """Finds all tracks before or after a division."""
    for track in experiment.links.find_all_tracks():
        if track.will_divide() or _has_divided(track):
            yield track


def _has_divided(track: LinkingTrack) -> bool:
    previous_tracks = track.get_previous_tracks()
    if len(previous_tracks) == 1 and previous_tracks.pop().will_divide():
        return True
    return False


def _plot_intensities_by_cell_cycle(window: Window):
    intensities_by_name = defaultdict(_IntensitiesOverTime)
    for experiment in window.get_active_experiments():

        # We will iterate multiple times over this generator, so put it in a list
        cell_cycle_tracks = list(_find_tracks_connected_to_division(experiment))

        timings = experiment.images.timings()

        for intensity_key in intensity_calculator.get_intensity_keys(experiment):
            for track in cell_cycle_tracks:
                delta_times_with_start_h = list()
                delta_time_with_end_h = list()
                intensities = list()

                previous_division_h = timings.get_time_h_since_start(track.first_time_point_number() - 1)
                next_division_h = timings.get_time_h_since_start(track.last_time_point_number())
                division_before = _has_divided(track)
                division_after = track.will_divide()

                for position in track.positions():
                    intensity = intensity_calculator.get_normalized_intensity(experiment, position,
                                                                              intensity_key=intensity_key,
                                                                              per_pixel=True)
                    if intensity is None:
                        continue

                    time_h = timings.get_time_h_since_start(position.time_point_number())
                    intensities.append(intensity)
                    delta_times_with_start_h.append(time_h - previous_division_h)
                    delta_time_with_end_h.append(time_h - next_division_h)

                # Add intensities of track
                if len(intensities) >= 2:
                    if division_before:
                        intensities_by_name[intensity_key].add_line(delta_times_with_start_h, intensities)
                    if division_after:
                        intensities_by_name[intensity_key].add_line(delta_time_with_end_h, intensities)

    dialog.popup_figure(window, lambda figure: _draw_intensities_by_cell_cycle(figure, intensities_by_name))
