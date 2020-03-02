from typing import Dict, Any, Type

from matplotlib.backend_bases import KeyEvent

from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import activate, Visualizer
from organoid_tracker.visualizer.abstract_image_visualizer import AbstractImageVisualizer
from organoid_tracker.visualizer.standard_image_visualizer import StandardImageVisualizer


class ExitableImageVisualizer(AbstractImageVisualizer):

    _parent_viewer: Type[Visualizer]  # When exiting this viewer, the viewer specified here is opened.

    def __init__(self, window: Window, parent_viewer: Type[Visualizer] = StandardImageVisualizer):
        """Creates this viewer. parent_viewer is opened when you exit this viewer."""
        super().__init__(window)
        self._parent_viewer = parent_viewer

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "View//Exit-Exit this view [Esc]": self._exit_view,
        }

    def _exit_view(self):
        from organoid_tracker.visualizer.standard_image_visualizer import StandardImageVisualizer
        image_visualizer = self._parent_viewer(self._window)
        activate(image_visualizer)

    def _on_command(self, command: str) -> bool:
        if command == "help":
            self.update_status("Available commands:"
                               "  /t20: Jump to time point 20 (also works for other time points)"
                               "  /exit: Exits this view")
            return True
        if command == "exit":
            self._exit_view()
            return True
        return super()._on_command(command)
