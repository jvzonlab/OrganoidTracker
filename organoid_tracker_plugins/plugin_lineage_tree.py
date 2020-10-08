from typing import Dict, Any, Tuple, Set, Optional

from matplotlib.backend_bases import MouseEvent
import matplotlib.colors
import matplotlib.cm

from organoid_tracker.core import UserError, Color
from organoid_tracker.core.links import LinkingTrack, Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.location_map import LocationMap
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import linking_markers, lineage_markers
from organoid_tracker.linking_analysis.lineage_division_counter import get_min_division_count_in_lineage
from organoid_tracker.linking_analysis.lineage_drawing import LineageDrawing
from organoid_tracker.linking_analysis.linking_markers import EndMarker
from organoid_tracker.visualizer import Visualizer


_SPLINE_COLORMAP = matplotlib.cm.get_cmap("YlOrBr")


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Lineages-Interactive lineage tree...": lambda: _show_lineage_tree(window)
    }


def _show_lineage_tree(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")

    dialog.popup_visualizer(window.get_gui_experiment(), LineageTreeVisualizer)


class LineageTreeVisualizer(Visualizer):
    """Shows lineage trees. Double-click somewhere to go to that location in the microscopy data."""

    _location_map: Optional[LocationMap] = None
    _track_to_manual_color: Dict[LinkingTrack, Color]

    _min_division_count: int = 0
    _display_deaths: bool = True
    _display_warnings: bool = True
    _display_manual_colors: bool = False
    _display_cell_type_colors: bool = True
    _display_axis_colors: bool = False
    _display_track_id: bool = False

    def __init__(self, window: Window):
        super().__init__(window)
        self._track_to_manual_color = dict()

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            "View//Toggles-Toggle showing warnings": self._toggle_errors,
            "View//Toggles-Toggle showing deaths": self._toggle_deaths,
            "View//Toggles-Toggle showing lineage ids": self._toggle_track_id,
            "View//Color toggles-Toggle showing cell types": self._toggle_cell_types,
            "View//Color toggles-Toggle showing manual colors": self._toggle_manual_colors,
            "View//Color toggles-Toggle showing axis positions": self._toggle_axis_colors,
            "View//Divisions-Require X amount of divisions...": self._set_minimum_divisions
        }

    def _uncolor_lineages(self):
        self._display_cell_type_colors = False
        self._display_axis_colors = False
        self._display_manual_colors = False

    def _toggle_deaths(self):
        self._display_deaths = not self._display_deaths
        self.draw_view()

    def _toggle_errors(self):
        self._display_warnings = not self._display_warnings
        self.draw_view()

    def _toggle_cell_types(self):
        self._display_cell_type_colors = not self._display_cell_type_colors

        if self._display_cell_type_colors:
            self._uncolor_lineages()
            self._display_cell_type_colors = True
            self.update_status("Now coloring by cell type; turned off other lineage coloring")
        else:
            self.update_status("No longer coloring by cell type")
        self.draw_view()

    def _toggle_manual_colors(self):
        self._display_manual_colors = not self._display_manual_colors
        if self._display_manual_colors:
            self._uncolor_lineages()
            self._display_manual_colors = True
            self.update_status("Now coloring by manually assigned colors; turned off other lineage coloring")
        else:
            self.update_status("No longer coloring by manually assigned colors")
        self.draw_view()

    def _toggle_axis_colors(self):
        self._display_axis_colors = not self._display_axis_colors
        if self._display_axis_colors:
            self._uncolor_lineages()
            self._display_axis_colors = True
            self.update_status("Now coloring by axis position; turned off other lineage coloring")
        else:
            self.update_status("No longer coloring by axis position")
        self.draw_view()

    def _toggle_track_id(self):
        self._display_track_id = not self._display_track_id
        if self._display_track_id:
            self.update_status("Now displaying lineage ids")
        else:
            self.update_status("No longer displaying lineage ids")
        self.draw_view()

    def _get_track_label(self, track: LinkingTrack) -> Optional[str]:
        """Gets the label that should be displayed for the given track."""
        if self._display_track_id:
            return str(self._experiment.links.get_track_id(track))
        return None

    def _set_minimum_divisions(self):
        min_division_count = dialog.prompt_int("Minimum division count",
                                               "How many divisions need to happen in a lineage"
                                               " tree before it shows up?", minimum=0, default=self._min_division_count)
        if min_division_count is None:
            return
        self._min_division_count = min_division_count
        self.draw_view()

    def _calculate_track_colors(self):
        """Places the manually assigned lineage colors in a track-to-color map."""
        self._track_to_manual_color.clear()

        links = self._experiment.links
        for track in links.find_starting_tracks():
            next_tracks = track.get_next_tracks()
            if len(next_tracks) == 0:
                continue  # No colors for tracks without divisions
            else:
                color = lineage_markers.get_color(links, track)
                self._give_lineage_color(track, color)

    def _calculate_axis_positions_if_enabled(self) -> Tuple[Dict[Position, float], float]:
        """Calculates, for all positions in the experiment, the position on the axis."""
        all_positions = dict()
        max_position = 0.001
        if not self._display_axis_colors:
            # Skip rest of method, coloring this way is disabled.
            return all_positions, max_position

        experiment = self._experiment
        splines = experiment.splines
        positions = experiment.positions
        for time_point in positions.time_points():
            for position in positions.of_time_point(time_point):
                axis_position = splines.to_position_on_original_axis(experiment.links, position)
                if axis_position is None:
                    continue
                axis_position_float = axis_position.pos
                all_positions[position] = axis_position_float
                max_position = max(max_position, axis_position_float)

        return all_positions, max_position

    def _give_lineage_color(self, linking_track: LinkingTrack, color: Color):
        """Gives a while lineage (including all children) a color."""
        self._track_to_manual_color[linking_track] = color
        for next_track in linking_track.get_next_tracks():
            self._give_lineage_color(next_track, color)

    def draw_view(self):
        self._clear_axis()

        experiment = self._experiment
        position_data = experiment.position_data
        links = experiment.links
        links.sort_tracks_by_x()

        self._calculate_track_colors()
        axis_positions, highest_axis_position = self._calculate_axis_positions_if_enabled()
        print(axis_positions, highest_axis_position)

        def color_getter(time_point_number: int, track: LinkingTrack) -> Tuple[float, float, float]:
            if self._display_warnings:
                if _has_error_close_in_time(position_data, time_point_number, track):
                    return 0.7, 0.7, 0.7

            if self._display_deaths and track.max_time_point_number() - time_point_number < 10:
                end_marker = linking_markers.get_track_end_marker(position_data, track.find_last_position())
                if end_marker == EndMarker.DEAD:
                    return 1, 0, 0
                elif EndMarker.is_shed(end_marker):
                    return 0, 0, 1
            if self._display_manual_colors:
                color = self._track_to_manual_color.get(track)
                if color is not None:
                    return color.to_rgb_floats()
            if self._display_axis_colors:
                axis_position = axis_positions.get(track.find_position_at_time_point_number(time_point_number))
                if axis_position is not None:
                    color_fraction = axis_position / highest_axis_position * 0.8 + 0.2
                    return _SPLINE_COLORMAP(color_fraction)
            if self._display_cell_type_colors:
                color = self._get_type_color(track.find_position_at_time_point_number(time_point_number))
                if color is not None:
                    return matplotlib.colors.to_rgb(color)
            return 0, 0, 0  # Default is black

        def lineage_filter(linking_track: LinkingTrack) -> bool:
            if self._min_division_count <= 0:
                return True  # Don't even check, every lineage has 0 or more divisions
            return get_min_division_count_in_lineage(linking_track) >= self._min_division_count

        resolution = ImageResolution(1, 1, 1, 60)
        self._location_map = LocationMap()
        width = LineageDrawing(links).draw_lineages_colored(self._ax, color_getter=color_getter,
                                                            resolution=resolution,
                                                            location_map=self._location_map,
                                                            lineage_filter=lineage_filter,
                                                            label_getter=self._get_track_label)

        self._ax.set_ylabel("Time (time points)")
        if self._ax.get_xlim() == (0, 1):
            # Only change axis if the default values were used
            self._ax.set_ylim([experiment.last_time_point_number(), experiment.first_time_point_number() - 1])
            self._ax.set_xlim([-0.1, width + 0.1])

        self._fig.canvas.draw()

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return
        if self._location_map is None:
            return
        position: Optional[Position] = self._location_map.get_nearby(event.xdata, event.ydata)
        if position is None:
            return
        self.get_window().get_gui_experiment().goto_position(position)
        self.update_status("Focused main window on " + str(position))


def _has_error_close_in_time(position_data: PositionData, time_point_number: int, track: LinkingTrack, time_window: int = 5):
    min_t = max(track.min_time_point_number(), time_point_number - time_window)
    max_t = min(track.max_time_point_number(), time_point_number + time_window)
    for t in range(min_t, max_t + 1):
        if linking_markers.get_error_marker(position_data, track.find_position_at_time_point_number(t)):
            return True
    return False
