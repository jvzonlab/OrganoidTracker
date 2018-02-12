import imaging
from imaging import Experiment, Particle
from matplotlib.figure import Figure, Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from typing import Iterable, Optional
import matplotlib.pyplot as plt


class Visualizer:
    """A complete application for visualization of an experiment"""
    _experiment: Experiment

    _fig: Figure
    _ax: Axes

    _key_handler_id: int
    _mouse_handler_id: int
    _pending_text: Optional[str]

    def __init__(self, experiment: Experiment, figure: Figure):
        self._experiment = experiment
        self._fig = figure
        self._ax = self._fig.gca()
        self._key_handler_id = self._fig.canvas.mpl_connect("key_press_event", self._on_key_press_raw)
        self._mouse_handler_id = self._fig.canvas.mpl_connect("button_press_event", self._on_mouse_click)
        self._pending_text = None

    def _clear_axis(self):
        """Clears the axis, except that zoom settings are preserved"""
        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        self._ax.clear()
        if xlim[1] - xlim[0] > 2:
            # Only preserve scale if some sensible value was recorded
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            self._ax.set_autoscale_on(False)

    def draw_view(self):
        print("Override this method to draw the view.")

    def _on_key_press_raw(self, event: KeyEvent):
        # Records commands
        if event.key == '/':
            # Entering command mode
            self._pending_text = ""
            return

        if self._pending_text is None:
            self._on_key_press(event)
        else:
            if event.key == 'enter':
                # Finish typing command
                text = self._pending_text
                self._pending_text = None
                if len(text) > 0:
                    self._on_command(text)
            else:
                self._pending_text += event.key

    def _on_key_press(self, event: KeyEvent):
        pass

    def _on_command(self, text: str):
        pass

    def _on_mouse_click(self, event: MouseEvent):
        pass

    def detach(self):
        self._fig.canvas.mpl_disconnect(self._key_handler_id)
        self._fig.canvas.mpl_disconnect(self._mouse_handler_id)

    @staticmethod
    def get_closest_particle(particles: Iterable[Particle], x: int, y: int, z: Optional[int], max_distance: int = 100000):
        """Gets the particle closest ot the given position."""
        search_position = Particle(x, y, z)
        return imaging.get_closest_particle(particles, search_position, ignore_z=z is None, max_distance=max_distance)


_visualizer = None # Reference to prevent event handler from being garbage collected


def _configure_matplotlib():
    plt.rcParams['keymap.forward'] = []
    plt.rcParams['keymap.back'] = ['backspace']
    plt.rcParams['keymap.fullscreen'] = ['ctrl+f']
    plt.rcParams['keymap.save'] = ['ctrl+s']


def activate(visualizer: Visualizer) -> None:
    _configure_matplotlib()

    global _visualizer
    if _visualizer is not None:
        # Unregister old event handlers
        _visualizer.detach()

    _visualizer = visualizer
    _visualizer.draw_view()

