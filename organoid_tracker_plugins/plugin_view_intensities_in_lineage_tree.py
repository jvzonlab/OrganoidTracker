from functools import partial
from typing import Dict, Any, Optional, List

import matplotlib
import numpy
from matplotlib.colors import Colormap

from organoid_tracker.core import UserError, Color
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.visualizer.lineage_tree_visualizer import LineageTreeVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//View-Intensity-colored lineage tree...": lambda: _show_lineage_tree(window)
    }


def _show_lineage_tree(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")
    intensity_keys = intensity_calculator.get_intensity_keys(experiment)
    if len(intensity_keys) == 0:
        raise UserError("No intensities recorded", "No intensities were recorded. Cannot plot anything.")

    dialog.popup_visualizer(window, IntensityLineageTreeVisualizer)


class IntensityLineageTreeVisualizer(LineageTreeVisualizer):
    """Shows lineage trees colored by the intensity of the positions."""

    _intensity_key: str
    _intensity_min_value: float = 0.0
    _intensity_max_value: float = 1.0
    _intensity_scaling_quantile: float = 0
    _intensity_colormap: Colormap = matplotlib.cm.coolwarm
    _intensity_nan_color: Color = Color.black()
    _intensity_sorting_key: Optional[str] = None  # Sorts the tracks using this intensity

    def __init__(self, window: Window):
        super().__init__(window)

        # Set default intensity key
        intensity_keys = intensity_calculator.get_intensity_keys(self._experiment)
        if len(intensity_keys) > 0:
            self._intensity_key = intensity_keys[0]
        else:
            self._intensity_key = intensity_calculator.DEFAULT_INTENSITY_KEY

        # Default settings
        self._calculate_min_max_intensity()
        self._uncolor_lineages()
        self._display_deaths = False
        self._display_custom_colors = True

    def _calculate_min_max_intensity(self):
        experiment = self._experiment

        if self._intensity_scaling_quantile == 0:
            # Just find min and max (should be faster than building a list and then sorting it)
            min_value = None
            max_value = None

            for time_point in experiment.positions.time_points():
                for position in experiment.positions.of_time_point(time_point):
                    intensity = intensity_calculator.get_normalized_intensity(self._experiment, position,
                                                                              intensity_key=self._intensity_key)
                    if intensity is None:
                        continue
                    if min_value is None or intensity < min_value:
                        min_value = intensity
                    if max_value is None or intensity > max_value:
                        max_value = intensity

            if min_value is None or max_value is None:
                self._intensity_min_value = 0
                self._intensity_max_value = 1
            else:
                self._intensity_min_value = min_value
                self._intensity_max_value = max_value
        else:
            # Collect all intensities to find the number at the right quantile
            intensities = list()
            for time_point in experiment.positions.time_points():
                for position in experiment.positions.of_time_point(time_point):
                    intensity = intensity_calculator.get_normalized_intensity(self._experiment, position,
                                                                              intensity_key=self._intensity_key)
                    if intensity is not None:
                        intensities.append(intensity)
            if len(intensities) < 2:
                self._intensity_min_value = 0
                self._intensity_max_value = 1
            else:
                intensities = numpy.array(intensities, numpy.float32)
                self._intensity_min_value = numpy.quantile(intensities, q=self._intensity_scaling_quantile)
                self._intensity_max_value = numpy.quantile(intensities, q=1 - self._intensity_scaling_quantile)


    def _get_custom_color_label(self) -> Optional[str]:
        return "intensities"

    def _get_lineage_line_width(self) -> float:
        return 3

    def get_extra_menu_options(self) -> Dict[str, Any]:
        intensity_keys = intensity_calculator.get_intensity_keys(self._experiment)
        if len(intensity_keys) == 0:
            intensity_keys = [intensity_calculator.DEFAULT_INTENSITY_KEY]

        options = {
            **super().get_extra_menu_options(),
            "Intensity//Colors-Colormap//Split-Blue to red": partial(self._set_colormap, "coolwarm"),
            "Intensity//Colors-Colormap//Split-Pink to green": partial(self._set_colormap, "PiYG"),
            "Intensity//Colors-Colormap//Split-Purple to green": partial(self._set_colormap, "PRGn"),
            "Intensity//Colors-Colormap//Single-Blue": partial(self._set_colormap, "Blues"),
            "Intensity//Colors-Colormap//Single-Grey": partial(self._set_colormap, "Greys"),
            "Intensity//Colors-Colormap//Single-Green": partial(self._set_colormap, "Greens"),
            "Intensity//Colors-Colormap//Single-Orange": partial(self._set_colormap, "Oranges"),
            "Intensity//Colors-Colormap//Single-Purple": partial(self._set_colormap, "Purples"),
            "Intensity//Colors-Colormap//Single-Red": partial(self._set_colormap, "Reds"),
            "Intensity//Colors-Colormap//Uniform-Cividis": partial(self._set_colormap, "cividis"),
            "Intensity//Colors-Colormap//Uniform-Inferno": partial(self._set_colormap, "inferno"),
            "Intensity//Colors-Colormap//Uniform-Magma": partial(self._set_colormap, "magma"),
            "Intensity//Colors-Colormap//Uniform-Plasma": partial(self._set_colormap, "plasma"),
            "Intensity//Colors-Colormap//Uniform-Viridis": partial(self._set_colormap, "viridis"),
            "Intensity//Colors-Color scaling//Scale from min to max": self._set_scaling_minmax,
            "Intensity//Colors-Color scaling//Scale using quantile...": self._prompt_scaling_quantile,
        }
        if len(intensity_keys) > 1:
            for intensity_key in intensity_keys:
                options["Intensity//Intensity selector//" + intensity_key] = partial(self._switch_intensity_key, intensity_key)
                options["Sort//Sort tracks by intensity//" + intensity_key] = partial(self._switch_sorting_key, intensity_key)
        else:
            options["Sort//Sort tracks by intensity"] = partial(self._switch_sorting_key, intensity_keys[0])
        options["Sort//Sort tracks by x"] = partial(self._switch_sorting_key, None)

        return options

    def _set_scaling_minmax(self):
        self._intensity_scaling_quantile = 0
        self._calculate_min_max_intensity()
        self.draw_view()
        self.update_status("Now scaling from overall minimum to maximum.")

    def _prompt_scaling_quantile(self):
        quantile = dialog.prompt_float("Scaling", "What scaling quantile should we use? A scaling quantile of 0.1 means"
                                        " that we scale from the 10% darkest to the 10% brightest", minimum=0,
                                       maximum=0.49, decimals=2, default=self._intensity_scaling_quantile)
        if quantile is None:
            return
        self._intensity_scaling_quantile = quantile
        self._calculate_min_max_intensity()
        self.draw_view()
        if self._intensity_scaling_quantile == 0:
            self.update_status("Now scaling from overall minimum to maximum.")
        else:
            self.update_status(f"Now scaling from {quantile * 100}% darkest to {100 - quantile * 100}% values accross all time points.")

    def _switch_sorting_key(self, sorting_key: Optional[str]):
        self._intensity_sorting_key = sorting_key
        self.draw_view()
        if sorting_key is None:
            self.update_status("Now sorting by track x position, so the leftmost track in the organoid appears on the left.")
        else:
            self.update_status("Now sorting by the intensity stored under the \"" + sorting_key + "\" key.")


    def _set_colormap(self, name: str):
        self._intensity_colormap = matplotlib.colormaps.get(name)
        self.update_status(f"Now coloring by the \"{name}\" colormap of Matplotlib.")
        self.draw_view()

    def _switch_intensity_key(self, intensity_key: str):
        self._uncolor_lineages()
        self._intensity_key = intensity_key
        self._calculate_min_max_intensity()
        self._display_custom_colors = True
        self.update_status(f"Now coloring by the intensity values stored under \"{intensity_key}\"; "
                           f"turned off other lineage coloring")
        self.draw_view()

    def refresh_data(self):
        self._calculate_min_max_intensity()
        self.draw_view()

    def _get_sorted_tracks(self) -> List[LinkingTrack]:
        if self._intensity_sorting_key is None:
            # Sort by track x
            links = self._experiment.links
            links.sort_tracks_by_x()
            return list(links.find_starting_tracks())

        # Sort by intensity
        starting_tracks = list(self._experiment.links.find_starting_tracks())

        key = self._intensity_sorting_key
        experiment = self._experiment
        intensity_by_track = dict()
        for starting_track in starting_tracks:
            starting_track_with_daughters = {starting_track} | starting_track.get_next_tracks()
            intensities = list()
            for track in starting_track_with_daughters:
                for position in track.positions():
                    intensity = intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=key)
                    if intensity is not None:
                        intensities.append(intensity)
            if len(intensities) > 0:
                intensity_by_track[starting_track] = sum(intensities) / len(intensities)
            else:
                intensity_by_track[starting_track] = 0

        starting_tracks.sort(key=lambda t: intensity_by_track[t], reverse=True)
        return starting_tracks

    def _get_custom_color(self, position: Position) -> Optional[Color]:
        intensity = intensity_calculator.get_normalized_intensity(self._experiment, position,
                                                                  intensity_key=self._intensity_key)
        if intensity is None or self._intensity_min_value == self._intensity_max_value:
            return self._intensity_nan_color

        intensity = (intensity - self._intensity_min_value) / (self._intensity_max_value - self._intensity_min_value)
        if intensity < 0:
            intensity = 0.0
        if intensity > 1:
            intensity = 1.0
        r, g, b, a = self._intensity_colormap(intensity)
        return Color(int(r * 255), int(g * 255), int(b * 255))
