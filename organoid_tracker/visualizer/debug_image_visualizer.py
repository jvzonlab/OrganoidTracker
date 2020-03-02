from matplotlib import cm
from matplotlib.colors import Colormap
from numpy import ndarray

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui.launcher import launch_window, mainloop
from organoid_tracker.gui.window import Window
from organoid_tracker.image_loading.array_image_loader import SingleImageLoader
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.abstract_image_visualizer import AbstractImageVisualizer


def popup_3d_image(image: ndarray, name: str, cmap: Colormap = cm.get_cmap("gray")):
    """To be used in scripts, where no GUI loop is running. After showing the image, the program exits."""
    experiment = Experiment()
    experiment.images.image_loader(SingleImageLoader(image))
    window = launch_window(experiment)
    visualizer = _DebugImageVisualizer(window, name, cmap)
    activate(visualizer)
    mainloop()


class _DebugImageVisualizer(AbstractImageVisualizer):

    _name: str

    def __init__(self, window: Window, name: str, cmap: Colormap):
        super().__init__(window)
        self._name = name
        self._color_map = cmap

    def _get_figure_title(self) -> str:
        return self._name + " (z=" + str(self._z) + ")"

