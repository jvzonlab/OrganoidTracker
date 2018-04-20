import numpy

import imaging
from gui import Window
from imaging import Experiment, Particle, TimePoint
from matplotlib.figure import Figure, Axes, Text
from matplotlib.backend_bases import KeyEvent, MouseEvent
from typing import Iterable, Optional, Union, Dict, List
import matplotlib.pyplot as plt


class Visualizer:
    """A complete application for visualization of an experiment"""
    _experiment: Experiment
    _window: Window
    _fig: Figure
    _ax: Axes

    _pending_command_text: Optional[str]

    def __init__(self, window: Window):
        if not isinstance(window, Window):
            raise ValueError("window is not a Window")
        self._experiment = window.get_experiment()
        self._window = window
        self._fig = window.get_figure()
        self._ax = self._fig.gca()

        self._pending_command_text = None

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

    def draw_view(self):
        raise NotImplementedError()

    def update_status(self, text: Union[str, bytes], redraw=True):
        self._window.set_status(str(text))

    def _on_key_press_raw(self, event: KeyEvent):
        # Records commands

        if self._pending_command_text is None:
            if event.key == '/':
                # Staring command mode
                self._pending_command_text = ""
                self.update_status("/")
                return
            self._on_key_press(event)
        else:
            if event.key == 'enter':
                # Finish typing command
                text = self._pending_command_text
                self._pending_command_text = None
                if len(text) > 0:
                    if not self._on_command(text):
                        self.update_status("Unknown command: " + text + ". Type /help for help.")
            elif event.key == 'escape':
                # Exit typing command
                self._pending_command_text = None
                self.update_status(self.__doc__)
            else:
                if event.key == 'backspace':
                    if len(self._pending_command_text) > 0:
                        self._pending_command_text = self._pending_command_text[:-1]
                elif len(event.key) > 1:
                    pass  # Pressing "shift", "control", "left", etc.
                else:
                    self._pending_command_text += event.key
                self.update_status("/" + self._pending_command_text)

    def _on_key_press(self, event: KeyEvent):
        pass

    def _on_command(self, text: str) -> bool:
        return False

    def _on_mouse_click(self, event: MouseEvent):
        pass

    def attach(self):
        self._window.setup_menu(self.get_extra_menu_options())
        self._window.register_event_handler("key_press_event", self._on_key_press_raw)
        self._window.register_event_handler("button_press_event", self._on_mouse_click)

    def detach(self):
        self._window.unregister_event_handlers()

    def get_extra_menu_options(self) -> Dict[str, List]:
        return {}

    def create_image(self, time_point: TimePoint, show_next_time_point: bool):
        """Creates an image suitable for display purposes. IF show_next_time_point is set to True, then then a color
        image will be created with the next image in red, and the current image in green."""
        time_point_images = time_point.load_images()
        if time_point_images is None:
            return None
        if show_next_time_point:
            image_shape = time_point_images.shape

            rgb_images = numpy.zeros((image_shape[0], image_shape[1], image_shape[2], 3), dtype='float')
            rgb_images[:,:,:,1] = time_point_images  # Green channel is current image
            try:
                next_time_point = self._experiment.get_next_time_point(time_point)
                next_time_point_images = next_time_point.load_images()
                rgb_images[:,:,:,0] = next_time_point_images # Red channel is next image
            except KeyError:
                pass  # There is no next time point, ignore
            time_point_images = rgb_images / rgb_images.max()
        return time_point_images

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
        return imaging.get_closest_particle(particles, search_position, ignore_z=ignore_z, max_distance=max_distance)


__active_visualizer = None # Reference to prevent event handler from being garbage collected


def activate(visualizer: Visualizer) -> None:
    global __active_visualizer
    if __active_visualizer is not None:
        # Unregister old event handlers
        __active_visualizer.detach()

    __active_visualizer = visualizer
    __active_visualizer.attach()
    __active_visualizer.draw_view()
    __active_visualizer.update_status(__active_visualizer.__doc__)

