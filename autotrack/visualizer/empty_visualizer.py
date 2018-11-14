from autotrack.gui import Window
from autotrack.visualizer.image_visualizer import StandardImageVisualizer
from autotrack.visualizer import Visualizer, activate


class EmptyVisualizer(Visualizer):
    """Created a new, empty project. Load some images to get started."""

    def __init__(self, window: Window):
        super().__init__(window)

    def refresh_view(self):
        if self._experiment.first_time_point_number() is None:
            return  # Nothing to refresh

        # Switch to more appropriate viewer
        visualizer = StandardImageVisualizer(self._window)
        activate(visualizer)

    def draw_view(self):
        self._clear_axis()
        self._window.set_figure_title("Empty project. Use the File menu to import data.")
        self._fig.canvas.draw()

    def _get_window_title(self):
        return "New project"
