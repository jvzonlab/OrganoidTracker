from typing import Dict, Any

from matplotlib.backend_bases import KeyEvent

from autotrack.visualizer import activate
from autotrack.visualizer.abstract_image_visualizer import AbstractImageVisualizer


class ExitableImageVisualizer(AbstractImageVisualizer):

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "View/Exit-Exit this view (Esc)": self._exit_view,
        }

    def _exit_view(self):
        from autotrack.visualizer.standard_image_visualizer import StandardImageVisualizer
        image_visualizer = StandardImageVisualizer(self._window, self._time_point.time_point_number(), self._z,
                                                   self._display_settings)
        activate(image_visualizer)

    def _on_key_press(self, event: KeyEvent):
        if event.key == "escape":
            self._exit_view()
        else:
            super()._on_key_press(event)

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
