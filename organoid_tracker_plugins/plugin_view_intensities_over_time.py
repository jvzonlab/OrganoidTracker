from functools import partial
from typing import Optional, Dict, Any, Set, List

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from organoid_tracker.core import UserError, Color
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution, ImageTimings
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator
from organoid_tracker.util.moving_average import MovingAverage
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer

_AVERAGING_WINDOW_TIME_STEPS = 20
_STEP_SIZE_H = 0.2


def get_menu_items(window: Window) -> Dict[str, Any]:
    # Dynamic menu entries depending on the available intensities

    # Collect intensities
    intensity_keys = set()
    for experiment in window.get_active_experiments():
        intensity_keys |= set(intensity_calculator.get_intensity_keys(experiment))

    if len(intensity_keys) == 0:
        # Just show an error message if clicked
        return {
            "Intensity//View-View individual intensities over time...":
                lambda: dialog.popup_error("No intensities recorded",
                                           "No intensities were recorded. Please do so first")
        }
    elif len(intensity_keys) == 1:
        # Just use the only intensity
        return {
            "Intensity//View-View individual intensities over time...": lambda: _view_intensities(window, intensity_keys.pop())
        }
    else:
        # Separate menu option for each
        return_value = dict()
        for intensity_key in intensity_keys:
            return_value["Intensity//View-View individual intensities over time//" + intensity_key] \
                = partial(_view_intensities, window, intensity_key)
        return return_value


def _view_intensities(window: Window, intensity_key: str):
    activate(_IntensityOverTimePlotter(window, intensity_key))


class _Line:
    times_h: List[float]
    intensities: List[float]
    positions: List[Position]
    track_id: int

    def __init__(self, timings: ImageTimings, positions: List[Position], intensities: List[Optional[float]],
                 track_id: int):
        self.times_h = list()
        self.intensities = list()
        self.positions = list()
        for position, intensity in zip(positions, intensities):
            if intensity is not None:
                time_h = timings.get_time_h_since_start(position.time_point_number())
                self.times_h.append(time_h)
                self.intensities.append(intensity)
                self.positions.append(position)
        self.track_id = track_id

    def get_position_at(self, at_time_h: float) -> Optional[Position]:
        """Gets the positions closest in time to the given time."""
        closest_index = self._get_nearest_index(at_time_h)
        if closest_index is None:
            return None
        return self.positions[closest_index]

    def _get_nearest_index(self, at_time_h: float) -> Optional[int]:
        closest_time_dh = None
        closest_index = None
        for index, time_h in enumerate(self.times_h):
            time_dh = abs(time_h - at_time_h)
            if closest_time_dh is None or time_dh < closest_time_dh:
                closest_time_dh = time_dh
                closest_index = index

        return closest_index


class _PlotData:
    """Contains the data for the intensity over time plot."""
    _y_machine_name: str
    y_display_name: str
    _lines: List[_Line]

    def __init__(self, y_machine_name: str, y_display_name: str):
        self._y_machine_name = y_machine_name
        self.y_display_name = y_display_name
        self._lines = list()

    def add_line(self, timings: ImageTimings, positions: List[Position], y_values: List[Optional[float]],
                 track_id: int):
        """Adds the intensity line of a single cell."""
        line = _Line(timings, positions, y_values, track_id)
        if len(line.times_h) > 0:
            # Only add if there are full data points (neighbor connections, recorded intensities, etc)
            self._lines.append(line)

    def export(self) -> List[Dict[str, List]]:
        """For JSON export."""
        return [
            {
                "times_h": line.times_h,
                self._y_machine_name: line.intensities,
                "track_id": line.track_id
            }
            for line in self._lines
        ]

    def plot_lines(self, ax: Axes, averaging_window_h: float):
        for line in self._lines:
            if len(self._lines) < 3 or averaging_window_h <= 0:
                ax.plot(line.times_h, line.intensities, color="gray")  # For plotting raw data
            if averaging_window_h > 0:
                average = MovingAverage(line.times_h, line.intensities, window_width=averaging_window_h,
                                        x_step_size=_STEP_SIZE_H)
                if len(average.x_values) < 1:
                    continue
                ax.text(average.x_values[-1], average.mean_values[-1], f"Track {line.track_id}", fontsize=6)
                average.plot(ax, label=None, color=Color(9, 132, 227))


class _IntensityOverTimePlotter(ExitableImageVisualizer):
    """Double-click on a cell to view the intensities over time."""

    _selected_tracks: Set[LinkingTrack]
    _intensity_key: str

    def __init__(self, window: Window, intensity_key: str):
        super().__init__(window)
        self._selected_tracks = set()
        self._experiment.links.sort_tracks_by_x()
        self._intensity_key = intensity_key

    def _get_figure_title(self) -> str:
        return "Time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) \
            + ", measuring " + self._intensity_key + ")"

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Graph//Intensities-Plot intensities...": self._plot_total_intensities,
            "Graph//Intensities-Plot raw intensities...": self._plot_raw_intensities,
            "Graph//Intensities-Plot measurement volume...": self._plot_volumes
        }

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        links = self._experiment.links
        track = links.get_track(position)
        if track is not None and track in self._selected_tracks:
            self._draw_selection(position, color)
            if dt == 0:
                self._ax.annotate(str(links.get_track_id(track)), (position.x, position.y),
                                  fontsize=8 - abs(dz / 2),
                                  fontweight="bold", color="black", backgroundcolor=(1, 1, 1, 0.8))
        return True

    def _on_mouse_single_click(self, event: MouseEvent):
        selected_position = self._get_position_at(event.xdata, event.ydata)
        if selected_position is None:
            self._selected_tracks.clear()
            self.draw_view()
            self.update_status("Deselected all tracks.")
            return

        track = self._experiment.links.get_track(selected_position)
        if track is None:
            self.update_status("Cannot select this position - it has no links.")
            return

        if track in self._selected_tracks:
            for some_track in track.find_all_previous_and_descending_tracks(include_self=True):
                if some_track in self._selected_tracks:
                    self._selected_tracks.remove(some_track)
            self.draw_view()
            self.update_status("Deselected this track.")
        else:
            for some_track in track.find_all_previous_and_descending_tracks(include_self=True):
                self._selected_tracks.add(some_track)
            self.draw_view()
            self.update_status("Selected this track.")

    def _plot_total_intensities(self):
        experiment = self._experiment
        timings = experiment.images.timings()
        plot_data = _PlotData(y_display_name=self._intensity_key, y_machine_name=self._intensity_key)
        for track in self._selected_tracks:
            positions = list(track.positions(connect_to_previous_track=True))
            intensities = [
                intensity_calculator.get_normalized_intensity(experiment, position, intensity_key=self._intensity_key)
                for position in track.positions(connect_to_previous_track=True)]
            plot_data.add_line(timings, positions, intensities, self._experiment.links.get_track_id(track))

        self._plot_line(plot_data)

    def _plot_volumes(self):
        timings = self._experiment.images.timings()
        position_data = self._experiment.position_data
        if not position_data.has_position_data_with_name(self._intensity_key + "_volume"):
            raise UserError("No measurement volume", "The volume in which the value \"" + self._intensity_key
                            + "\" was measured, is not known. This is because we didn't find any data for \""
                            + self._intensity_key + "_volume\".")

        plot_data = _PlotData(y_display_name="Measurement volume (pixels)", y_machine_name="volumes")
        for track in self._selected_tracks:
            positions = list(track.positions(connect_to_previous_track=True))
            volumes = [position_data.get_position_data(position, self._intensity_key + "_volume")
                       for position in track.positions(connect_to_previous_track=True)]

            plot_data.add_line(timings, positions, volumes, self._experiment.links.get_track_id(track))

        self._plot_line(plot_data)

    def _plot_raw_intensities(self):
        timings = self._experiment.images.timings()
        position_data = self._experiment.position_data
        plot_data = _PlotData(y_display_name=self._intensity_key, y_machine_name="intensities")
        for track in self._selected_tracks:
            positions = list(track.positions(connect_to_previous_track=True))

            # Calculate both lists, then divide them
            intensities = [position_data.get_position_data(position, self._intensity_key)
                           for position in track.positions(connect_to_previous_track=True)]

            plot_data.add_line(timings, positions, intensities, self._experiment.links.get_track_id(track))

        self._plot_line(plot_data, raw_lines=True)

    def _plot_line(self, plot_data: _PlotData, *, raw_lines: bool = False):
        self._assert_intensities_and_selection()

        resolution = self._experiment.images.resolution()
        averaging_window_h = 0 if raw_lines else _AVERAGING_WINDOW_TIME_STEPS * resolution.time_point_interval_h

        def plot(figure: Figure):
            ax: Axes = figure.gca()
            plot_data.plot_lines(ax, averaging_window_h=averaging_window_h)
            ax.set_xlabel("Time (h)")
            ax.set_ylabel(plot_data.y_display_name)

        def export():
            return {
                "tracks": plot_data.export()
            }

        dialog.popup_figure(self.get_window().get_gui_experiment(), plot, export_function=export,
                            size_cm=(11, 6))

    def _assert_intensities_and_selection(self):
        """Throws UserError if there are no intensities, or if no position has been selected."""
        if not self._experiment.position_data.has_position_data_with_name(self._intensity_key):
            raise UserError("No intensities recorded", "No intensities are recorded.\n"
                                                       "Please go back to the main screen and\n"
                                                       "record some intensities first.")
        if len(self._selected_tracks) == 0:
            raise UserError("No selected tracks", "No tracks are selected.\nPlease double"
                                                  "-click one or more tracks first.")
