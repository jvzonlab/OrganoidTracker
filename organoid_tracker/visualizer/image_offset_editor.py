from typing import Optional, Dict, Any

from matplotlib.backend_bases import KeyEvent

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import ImageOffsets
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.position_detection import position_mover
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


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


class ImageOffsetEditor(ExitableImageVisualizer):
    """Editor to add information on image offset, so that the object of interest can be kept at a fixed position.
    Use the WASD, Q and E keys to move all following images. Hold the ALT key for more fine-grained movement."""

    _previous_offsets: ImageOffsets

    def __init__(self, window: Window):
        window.display_settings.show_next_time_point = True
        super().__init__(window)

        self._previous_offsets = self._experiment.images.offsets.copy()

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            "Edit//Offsets-Reset all image offsets": self._reset_all_offsets
        }

    def _reset_all_offsets(self):
        self._experiment.images.offsets = ImageOffsets()
        self._exit_view()
        dialog.popup_message("Image offsets reset", "The image offsets of all time points have been reset to (0, 0, 0)."
                                                    " Press Ctrl+Z to undo.")

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
                               " is now " + str((new_offset.x, new_offset.y, new_offset.z)))
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
        from organoid_tracker.visualizer.link_and_position_editor import LinkAndPositionEditor
        data_editor = LinkAndPositionEditor(self._window)
        activate(data_editor)
