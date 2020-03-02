"""Used to color cells by their lineage."""
from typing import Dict, Any, Optional

from matplotlib.backend_bases import MouseEvent
import matplotlib.colors

from organoid_tracker.core import Color
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis import lineage_markers
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "View//Tracks-Cell colored by lineage...": lambda: _view_cells_colored_by_lineage(window)
    }


def _view_cells_colored_by_lineage(window: Window):
    activate(_CellsColoredByLineageVisualizer(window))


class _SetLineageColor(UndoableAction):
    _track: LinkingTrack
    _old_color: Color
    _new_color: Color

    def __init__(self, track: LinkingTrack, old_color: Color, new_color: Color):
        self._track = track
        self._old_color = old_color
        self._new_color = new_color

    def do(self, experiment: Experiment) -> str:
        lineage_markers.set_color(experiment.links, self._track, self._new_color)
        if self._new_color.is_black():
            return "Removed the color of the lineage"
        return f"Set the color of the lineage to {self._new_color}"

    def undo(self, experiment: Experiment) -> str:
        lineage_markers.set_color(experiment.links, self._track, self._old_color)
        if self._old_color.is_black():
            return "Removed the color of the lineage again"
        return f"Changed the color of the linage back to {self._old_color}"


class _CellsColoredByLineageVisualizer(ExitableImageVisualizer):
    """Colors each cell by its lineage: cells with the same color share a common ancestor. Double-click a cell to give
    it a color."""

    def __init__(self, window: Window):
        window.get_experiment().links.sort_tracks_by_x()
        super().__init__(window)

    def _on_mouse_click(self, event: MouseEvent):
        if event.xdata is None or event.ydata is None:
            return

        # Find clicked track
        position = self._get_position_at(event.xdata, event.ydata)
        if position is None:
            return

        from organoid_tracker.linking_analysis import lineage_id_creator
        links = self._experiment.links
        lineage_id = lineage_id_creator.get_lineage_id(links, position)
        track = links.get_track(position)
        if track is None:
            self.update_status(f"Selected {position} has no links, so it is not part of a lineage.")
            return

        color = lineage_markers.get_color(links, track)
        if event.dblclick:
            new_color = dialog.prompt_color("Choose a new color for the lineage", color)
            if new_color is not None:
                self.get_window().perform_data_action(_SetLineageColor(track, color, new_color))
        else:
            color_str = str(color) if not color.is_black() else "not specified"
            self.update_status(f"Lineage id of {position} is {lineage_id}. The color is {color_str}, double-click to"
                               f" edit.")

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        from organoid_tracker.linking_analysis import lineage_id_creator

        if abs(dz) > self.MAX_Z_DISTANCE or dt != 0:
            return super()._on_position_draw(position, color, dz, dt)

        links = self._experiment.links
        track = links.get_track(position)
        if track is None:
            return super()._on_position_draw(position, color, dz, dt)
        color = lineage_markers.get_color(links, track) if track is not None else None
        if color is None:
            color = Color.black()

        self._draw_selection(position, color.to_rgb_floats())
        return super()._on_position_draw(position, color, dz, dt)
