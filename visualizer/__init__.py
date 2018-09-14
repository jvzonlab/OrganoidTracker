"""A bunch of visualizers, all based on Matplotlib. (The fact that TkInter is also used is abstracted away.)"""

from typing import Iterable, Optional, Union, Dict, Any, Tuple

import numpy
from numpy import ndarray
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, Axes

import core
from core import Experiment, Particle, TimePoint
from gui import Window, dialog, Task


class DisplaySettings:
    show_next_time_point: bool
    show_images: bool
    show_reconstruction: bool

    def __init__(self, show_next_time_point: bool = False, show_images: bool = True, show_reconstruction: bool = False):
        self.show_next_time_point = show_next_time_point
        self.show_images = show_images
        self.show_reconstruction = show_reconstruction

    KEY_SHOW_NEXT_IMAGE_ON_TOP = "n"
    KEY_SHOW_IMAGES = "i"
    KEY_SHOW_RECONSTRUCTION = "r"


class Visualizer:
    """A complete application for visualization of an experiment"""
    _experiment: Experiment
    _window: Window
    _fig: Figure
    _ax: Axes

    def __init__(self, window: Window):
        if not isinstance(window, Window):
            raise ValueError("window is not a Window")
        self._experiment = window.get_experiment()
        self._window = window
        self._fig = window.get_figure()
        self._ax = self._fig.gca()

    def _clear_axis(self):
        """Clears the axis, except that zoom settings are preserved"""
        for image in self._ax.images:
            colorbar = image.colorbar
            if colorbar is not None:
                colorbar.remove()
        for text in self._fig.texts:
            text.remove()

        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        self._ax.clear()
        if xlim[1] - xlim[0] > 2:
            # Only preserve scale if some sensible value was recorded
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            self._ax.set_autoscale_on(False)

    def async(self, runnable, result_handler):
        """Creates a callable that runs the given runnable on a worker thread."""
        class MyTask(Task):
            def run(self):
                return runnable()

            def on_finished(self, result: Any):
                result_handler(result)

        def internal():
            self._window.get_scheduler().add_task(MyTask())
        return internal

    def draw_view(self):
        """Draws the view."""
        raise NotImplementedError()

    def refresh_view(self):
        """Draws the view, forcing to reload any cached data."""
        self.draw_view()

    def _update_status(self, text: Union[str, bytes], redraw=True):
        """Updates the status of the window."""
        self._window.set_status(str(text))

    def _on_key_press_raw(self, event: KeyEvent):
        """Calls _on_key_press, but catches all exceptions."""
        try:
            self._on_key_press(event)
        except Exception as e:
            dialog.popup_exception(e)

    def _on_key_press(self, event: KeyEvent):
        pass

    def _on_command_raw(self, text: str):
        """Executes a command, catches errors and shows a message if the command does not exist."""
        try:
            if not self._on_command(text):
                self._update_status("Unknown command: " + text)
        except Exception as e:
            dialog.popup_exception(e)

    def _on_command(self, text: str) -> bool:
        return False

    def _on_mouse_click(self, event: MouseEvent):
        pass

    def attach(self):
        self._window.setup_menu(self.get_extra_menu_options())
        self._window.set_window_title(self._get_window_title())
        self._window.register_event_handler("key_press_event", self._on_key_press_raw)
        self._window.register_event_handler("button_press_event", self._on_mouse_click)
        self._window.register_event_handler("refresh_event", self.refresh_view)
        self._window.register_event_handler("command_event", self._on_command_raw)

    def detach(self):
        self._window.unregister_event_handlers()

    def _get_window_title(self) -> Optional[str]:
        """Called to query what the window title should be. This will be prefixed with the name of the program."""
        return None

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {}

    def load_image(self, time_point: TimePoint, show_next_time_point: bool) -> Optional[ndarray]:
        """Creates an image suitable for display purposes. IF show_next_time_point is set to True, then then a color
        image will be created with the next image in red, and the current image in green."""
        time_point_images = self._experiment.get_image_stack(time_point)
        if time_point_images is None:
            return None
        if show_next_time_point:
            image_shape = time_point_images.shape

            rgb_images = numpy.zeros((image_shape[0], image_shape[1], image_shape[2], 3), dtype='float')
            rgb_images[:,:,:,1] = time_point_images  # Green channel is current image
            try:
                next_time_point = self._experiment.get_next_time_point(time_point)
                next_time_point_images = self._experiment.get_image_stack(next_time_point)
                rgb_images[:,:,:,0] = next_time_point_images # Red channel is next image
            except KeyError:
                pass  # There is no next time point, ignore
            time_point_images = rgb_images / rgb_images.max()
        return time_point_images

    def reconstruct_image(self, time_point: TimePoint, zyx_size: Tuple[int, int, int]) -> ndarray:
        rgb_images = numpy.zeros((zyx_size[0], zyx_size[1], zyx_size[2], 3), dtype='float')
        colors = [(1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        i = 0
        for particle, shape in time_point.particles_and_shapes().items():
            shape.draw3d_color(particle.x, particle.y, particle.z, 0, rgb_images, colors[i % len(colors)])
            i += 1
        rgb_images.clip(min=0, max=1, out=rgb_images)
        return rgb_images

    @staticmethod
    def get_closest_particle(particles: Iterable[Particle], x: Optional[int], y: Optional[int], z: Optional[int], max_distance: int = 100000):
        """Gets the particle closest ot the given position. If z is missing, it is ignored. If x or y are missing,
        None is returned.
        """
        if x is None or y is None:
            return None # Mouse outside figure, so x or y are None
        ignore_z = False
        if z is None:
            z = 0
            ignore_z = True
        search_position = Particle(x, y, z)
        return core.get_closest_particle(particles, search_position, ignore_z=ignore_z, max_distance=max_distance)

    def get_window(self):
        return self._window


__active_visualizer = None # Reference to prevent event handler from being garbage collected


def activate(visualizer: Visualizer) -> None:
    if visualizer.get_window().get_scheduler().has_active_tasks():
        raise core.UserError("Running a task", "Please wait until the current task has been finished before switching"
                                               " to another window.")
    global __active_visualizer
    if __active_visualizer is not None:
        # Unregister old event handlers
        __active_visualizer.detach()

    __active_visualizer = visualizer
    __active_visualizer.attach()
    __active_visualizer.draw_view()
    __active_visualizer._update_status(__active_visualizer.__doc__)

