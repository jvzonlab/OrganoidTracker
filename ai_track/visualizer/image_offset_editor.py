from typing import Optional

from matplotlib.backend_bases import KeyEvent

from ai_track.core import TimePoint
from ai_track.core.experiment import Experiment
from ai_track.core.images import ImageOffsets
from ai_track.gui.undo_redo import UndoableAction
from ai_track.gui.window import Window
from ai_track.position_detection import position_mover
from ai_track.visualizer import DisplaySettings, activate
from ai_track.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class _ChangeAllPositionsAction(UndoableAction):
    """We don't record individual undos for every time you press a WASD key - that would just be annoying. Instead, we
    take a snapshot of the offsets before and after, so that they can be done and undone."""

    _offsets_before: ImageOffsets
    _offsets_after: ImageOffsets

    def __init__(self, offsets_before: ImageOffsets, offsets_after: ImageOffsets):
        self._offsets_before = offsets_before
        self._offsets_after = offsets_after

    def do(self, experiment: Experiment) -> str:
        experiment.images.offsets = self._offsets_after
        position_mover.update_positions_for_changed_offsets(experiment, self._offsets_before)
        return "Moved all positions, links and images"

    def undo(self, experiment: Experiment) -> str:
        experiment.images.offsets = self._offsets_before
        position_mover.update_positions_for_changed_offsets(experiment, self._offsets_after)
        return "Moved all positions, links and images back"

def _showing_next_time_point(display_settings: Optional[DisplaySettings]) -> DisplaySettings:
    """Creates or modifies the display settings so that two time points are shown at once."""
    if display_settings is None:
        display_settings = DisplaySettings()
    display_settings.show_next_time_point = True
    return display_settings


class ImageOffsetEditor(ExitableImageVisualizer):
    """Editor to add information on image offset, so that the object of interest can be kept at a fixed position.
    Use the WASD, Q and E keys to move all following images. Hold the ALT key for more fine-grained movement."""

    _previous_offsets: ImageOffsets

    def __init__(self, window: Window, *, time_point: Optional[TimePoint] = None, z: int = 14,
                 display_settings: DisplaySettings = None):
        super().__init__(window, time_point=time_point, z=z, display_settings=_showing_next_time_point(display_settings))

        self._previous_offsets = self._experiment.images.offsets.copy()

    def _draw_positions(self):
        return  # Don't draw the positions - they are only noise, and we don't update them when moving the images

    def _on_key_press(self, event: KeyEvent):
        offset = None
        if event.key == "a":
            offset = (-10, 0, 0)
        elif event.key == "alt+a":
            offset = (-1, 0, 0)
        elif event.key == "d":
            offset = (10, 0, 0)
        elif event.key == "alt+d":
            offset = (1, 0, 0)
        elif event.key == "w":
            offset = (0, -10, 0)
        elif event.key == "alt+w":
            offset = (0, -1, 0)
        elif event.key == "s":
            offset = (0, 10, 0)
        elif event.key == "alt+s":
            offset = (0, 1, 0)
        elif event.key == "q":
            offset = (0, 0, 1)
        elif event.key == "e":
            offset = (0, 0, -1)

        if offset is not None:
            self._experiment.images.offsets.update_offset(*offset, self._time_point.time_point_number() + 1,
                                                          self._experiment.last_time_point_number())
            self._regenerate_image()
            new_offset = self._experiment.images.offsets.of_time_point(
                self._experiment.get_next_time_point(self._time_point))
            self.update_status("Updated the offset for this and all following time points. Offset of this time point"
                               " towards the next is now " + str((new_offset.x, new_offset.y, new_offset.z)))
        else:
            super()._on_key_press(event)

    def _regenerate_image(self):
        self._display_settings.show_next_time_point = True
        self._load_time_point(self._time_point)
        self.draw_view()

    def _exit_view(self):
        if self._experiment.images.offsets != self._previous_offsets:
            # Apply the offsets again, now also moving the positions and links, and using an UndoableAction
            current_offsets = self._experiment.images.offsets.copy()
            self.get_window().get_gui_experiment().undo_redo.do(
                _ChangeAllPositionsAction(self._previous_offsets, current_offsets), self._experiment)

        # Actually move
        from ai_track.visualizer.link_and_position_editor import LinkAndPositionEditor
        data_editor = LinkAndPositionEditor(self._window, time_point=self._time_point, z=self._z,
                                            display_settings=self._display_settings)
        activate(data_editor)
