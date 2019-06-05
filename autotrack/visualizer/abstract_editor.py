from matplotlib.backend_bases import KeyEvent

from autotrack.gui.undo_redo import UndoableAction
from autotrack.visualizer.exitable_image_visualizer import ExitableImageVisualizer


class AbstractEditor(ExitableImageVisualizer):

    def get_extra_menu_options(self):
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
        return "Editing time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"

    def _get_window_title(self) -> str:
        return "Manual data editing"

    def _perform_action(self, action: UndoableAction):
        self._experiment.images.resolution()  # Will trigger an exception early if no resolution was set

        status = self._window.get_undo_redo().do(action, self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)

    def _undo(self):
        status = self._window.get_undo_redo().undo(self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)

    def _redo(self):
        status = self._window.get_undo_redo().redo(self._experiment)
        self.get_window().redraw_data()
        self.update_status(status)
