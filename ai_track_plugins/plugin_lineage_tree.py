from typing import Dict, Any, Tuple, Set, Optional

from matplotlib.backend_bases import PickEvent, MouseEvent

from ai_track.core import UserError
from ai_track.core.links import LinkingTrack, Links
from ai_track.core.position import Position
from ai_track.core.resolution import ImageResolution
from ai_track.gui import dialog
from ai_track.gui.location_map import LocationMap
from ai_track.gui.window import Window
from ai_track.linking_analysis import linking_markers
from ai_track.linking_analysis.lineage_drawing import LineageDrawing
from ai_track.linking_analysis.linking_markers import EndMarker
from ai_track.visualizer import Visualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Graph//Lineages-Interactive lineage tree...": lambda: _show_lineage_tree(window)
    }


def _show_lineage_tree(window: Window):
    experiment = window.get_experiment()
    if not experiment.links.has_links():
        raise UserError("No links specified", "No links were loaded. Cannot plot anything.")

    dialog.popup_visualizer(window.get_gui_experiment(), LineageTreeVisualizer)


def _get_track_x(linking_track: LinkingTrack):
    return linking_track.find_first_position().x


class LineageTreeVisualizer(Visualizer):

    _location_map: Optional[LocationMap] = None

    def draw_view(self):
        self._clear_axis()

        experiment = self._experiment
        links = experiment.links
        links.sort_tracks(_get_track_x)

        tracks_with_errors = self._find_tracks_with_errors()

        def color_getter(time_point_number: int, track: LinkingTrack) -> Tuple[float, float, float]:
            if track in tracks_with_errors:
                return 0.7, 0.7, 0.7
            if track.max_time_point_number() - time_point_number < 10 and\
                    linking_markers.get_track_end_marker(links, track.find_last_position()) == EndMarker.DEAD:
                return 1, 0, 0
            return 0, 0, 0

        resolution = ImageResolution(1, 1, 1, 60)
        self._location_map = LocationMap()
        width = LineageDrawing(links).draw_lineages_colored(self._ax, color_getter, resolution, self._location_map)

        self._ax.set_ylabel("Time (time points)")
        if self._ax.get_xlim() == (0, 1):
            # Only change axis if the default values were used
            self._ax.set_ylim([experiment.last_time_point_number(), experiment.first_time_point_number() - 1])
            self._ax.set_xlim([-0.1, width + 0.1])

        self.update_status("Double-click somewhere in the lineage tree to jump to that cell. Note: this lineage tree updates live.")
        self._fig.canvas.draw()

    def _on_mouse_click(self, event: MouseEvent):
        if not event.dblclick:
            return
        if self._location_map is None:
            return
        position: Position = self._location_map.get_nearby(int(event.xdata), int(event.ydata))
        if position is None:
            return
        self.get_window().get_gui_experiment().goto_position(position)
        self.update_status("Focused main window on " + str(position))

    def _find_tracks_with_errors(self) -> Set[LinkingTrack]:
        links = self._experiment.links
        tracks_with_errors = set()
        for position in linking_markers.find_errored_positions(links):
            track = links.get_track(position)
            if track is not None:
                tracks_with_errors.add(track)
                for next_track in track.get_next_tracks():
                    tracks_with_errors.add(next_track)
        return tracks_with_errors
