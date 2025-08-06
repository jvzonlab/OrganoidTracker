from organoid_tracker.gui.window import Window
from organoid_tracker.visualizer import Visualizer, activate
from organoid_tracker.visualizer.standard_image_visualizer import StandardImageVisualizer


class EmptyVisualizer(Visualizer):
    """Created a new, empty project. Load some images to get started."""

    def __init__(self, window: Window):
        super().__init__(window)

    def refresh_data(self):
        self._exit_if_possible()

    def refresh_all(self):
        super().refresh_all()
        self._exit_if_possible()

    def _on_command(self, command: str) -> bool:
        if command == "exit":
            self.update_status("You're already in the home screen.")
            return True
        return False

    def _exit_if_possible(self):
        if self._experiment.first_time_point_number() is None:
            return  # Nothing to refresh

        # Switch to more appropriate viewer
        visualizer = StandardImageVisualizer(self._window)
        activate(visualizer)

    def _get_must_show_plugin_menus(self) -> bool:
        return True

    def draw_view(self):
        self._clear_axis()
        self._window.set_figure_title("Empty project. Get started by loading images and tracking data.")
        self._fig.canvas.draw_idle()

    def _get_window_title(self):
        return "New project"
