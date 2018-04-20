from gui import Window
from imaging.image_visualizer import StandardImageVisualizer
from imaging.visualizer import Visualizer, activate


class EmptyVisualizer(Visualizer):
    """Created a new, empty project. Load some images to get started."""

    def __init__(self, window: Window):
        super().__init__(window)

    def refresh_view(self):
        try:
            self._experiment.first_time_point_number()
        except ValueError:
            pass  # Nothing to refresh
        else:
            # Switch to more appropriate viewer
            visualizer = StandardImageVisualizer(self._window)
            activate(visualizer)

    def draw_view(self):
        self._clear_axis()
        self._window.set_title("Empty project. Use the File menu to import data.")
        self._fig.canvas.draw()
