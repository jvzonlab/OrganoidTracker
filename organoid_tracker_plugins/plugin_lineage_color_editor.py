"""Used to color cells by their lineage."""
from typing import Dict, Any, Optional, Tuple

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
from organoid_tracker.visualizer.abstract_editor import AbstractEditor
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Experiment-Recolor individual lineages...": lambda: _view_cells_colored_by_lineage(window)
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


class _GiveRandomLineageColor(UndoableAction):

    _old_colors: Dict[int, Color]
    _color_only_if_dividing: bool

    def __init__(self, *, color_only_if_dividing: bool):
        self._old_colors = dict()
        self._color_only_if_dividing = color_only_if_dividing

    def do(self, experiment: Experiment) -> str:
        from organoid_tracker.linking_analysis import lineage_id_creator
        self._old_colors = dict()  # Reset dictionary

        links = experiment.links
        links.sort_tracks_by_x()
        for starting_track in links.find_starting_tracks():
            color = Color.black()
            if not self._color_only_if_dividing or len(starting_track.get_next_tracks()) > 0:
                # Give a proper color
                color = lineage_id_creator.generate_color_for_lineage_id(links.get_track_id(starting_track))
            for track in starting_track.find_all_descending_tracks(include_self=True):
                self._old_colors[links.get_track_id(starting_track)] = lineage_markers.get_color(links, track)
                lineage_markers.set_color(links, track, color)
        if self._color_only_if_dividing:
            return "Changed all lineages with cell divisions to have a single, random color."
        return "Changed all lineages to have a single, random color."

    def undo(self, experiment: Experiment) -> str:
        links = experiment.links
        for track_id, track in links.find_all_tracks_and_ids():
            old_color = self._old_colors.get(track_id)
            if old_color is not None:
                lineage_markers.set_color(links, track, old_color)
        self._old_colors.clear()
        return "Restored the original color of all lineages."


class _RemoveColors(UndoableAction):
    _old_colors: Dict[Position, Color]

    def __init__(self):
        self._old_colors = dict()

    def do(self, experiment: Experiment) -> str:
        black = Color.black()
        links = experiment.links
        for starting_track in links.find_starting_tracks():
            old_color = lineage_markers.get_color(links, starting_track)
            if old_color != black:
                self._old_colors[starting_track.find_first_position()] = old_color
            lineage_markers.set_color(links, starting_track, black)

        return "Removed all colors. Don't worry, we have undo functionality."

    def undo(self, experiment: Experiment) -> str:
        links = experiment.links
        for position, color in self._old_colors.items():
            track = links.get_track(position)
            if track is not None:
                lineage_markers.set_color(links, track, color)
        self._old_colors.clear()  # Not needed anymore
        return "Restored the original color of all lineages."


class _CellsColoredByLineageVisualizer(AbstractEditor):
    """Colors each cell by its lineage: cells with the same color share a common ancestor. Double-click a cell to give
    it a color."""

    def __init__(self, window: Window):
        window.get_experiment().links.sort_tracks_by_x()
        super().__init__(window)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Color-Remove all colors": self._remove_all_colors,
            "Edit//Color-Randomize colors//All lineages": self._randomize_all_colors,
            "Edit//Color-Randomize colors//Dividing lineages only": self._randomize_dividing_colors
        }

    def _remove_all_colors(self):
        self._perform_action(_RemoveColors())

    def _randomize_all_colors(self):
        self._perform_action(_GiveRandomLineageColor(color_only_if_dividing=False))

    def _randomize_dividing_colors(self):
        self._perform_action(_GiveRandomLineageColor(color_only_if_dividing=True))

    def _on_mouse_single_click(self, event: MouseEvent):
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
        new_color = dialog.prompt_color("Choose a new color for the lineage", color)
        if new_color is not None:
            self._perform_action(_SetLineageColor(track, color, new_color))

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
