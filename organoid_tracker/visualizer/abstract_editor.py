from typing import Dict, Any

from matplotlib.backend_bases import KeyEvent

from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class AbstractEditor(ExitableImageVisualizer):

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Edit//Editor-Undo (Ctrl+z)": self._undo,
            "Edit//Editor-Redo (Ctrl+y)": self._redo,
        }

    def _on_key_press(self, event: KeyEvent):
        if event.key == "ctrl+z":
            self._undo()
        elif event.key == "ctrl+y":
            self._redo()
        else:
            super()._on_key_press(event)

    def _get_figure_title(self) -> str:
        return ("Editing time point " + str(self._time_point.time_point_number())
                + "    (z=" + self._get_figure_title_z_str() + ")")

    def _get_window_title(self) -> str:
        return "Manual data editing"

    def _perform_action(self, action: UndoableAction):
        self.get_window().perform_data_action(action)

    def _undo(self):
        status = self._window.get_undo_redo().undo(self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)

    def _redo(self):
        status = self._window.get_undo_redo().redo(self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)
