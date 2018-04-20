from gui import Window
from imaging import Experiment
from imaging.visualizer import Visualizer


class EmptyVisualizer(Visualizer):
    """Created a new, empty project. Load some images to get started."""

    def __init__(self, window: Window):
        super().__init__(window)

    def draw_view(self):
        self._clear_axis()
        self._window.set_title("Empty project")
        self._fig.canvas.draw()