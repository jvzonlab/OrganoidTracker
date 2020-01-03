"""A bunch of visualizers, all based on Matplotlib. (The fact that TkInter is also used is abstracted away.)"""
from typing import Iterable, Optional, Union, Dict, Any, Tuple

import numpy
from numpy import ndarray
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, Axes

from ai_track import core
from ai_track.core import TimePoint
from ai_track.core.position import Position
from ai_track.core.experiment import Experiment
from ai_track.core.resolution import ImageResolution
from ai_track.core.typing import MPLColor
from ai_track.gui import dialog
from ai_track.gui.threading import Task
from ai_track.gui.window import Window, DisplaySettings
from ai_track.imaging import cropper
from ai_track.linking.nearby_position_finder import find_closest_position
from ai_track.linking_analysis import linking_markers


class Visualizer:
    """A complete application for visualization of an experiment"""
    _window: Window
    _fig: Figure
    _ax: Axes
    _display_settings: DisplaySettings

    def __init__(self, window: Window):
        if not isinstance(window, Window):
            raise ValueError("window is not a Window")
        self._window = window
        self._fig = window.get_figure()
        self._ax = self._fig.gca()

    @property
    def _experiment(self) -> Experiment:
        """By making this a dynamic property (instead of just using self._experiment = window.get_experiment()), its
        value is always up-to-date, even if window.set_experiment(...) is called."""
        return self._window.get_experiment()

    @property
    def _time_point(self) -> TimePoint:
        """Gets the current time point that is visible. Only used in visualizers that display a single time point."""
        return self._window.display_settings.time_point

    @property
    def _z(self) -> int:
        """Gets the current z layer that is visible. Only used in the 2D image visualizers."""
        return self._window.display_settings.z

    @property
    def _display_settings(self) -> DisplaySettings:
        return self._window.display_settings

    def _clear_axis(self):
        """Clears the axis, except that zoom settings are preserved"""
        for image in self._ax.images:
            colorbar = image.colorbar
            if colorbar is not None:
                colorbar.remove_connection()
        for text in self._fig.texts:
            text.remove_connection()

        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        self._ax.clear()
        if xlim[1] - xlim[0] > 2:
            # Only preserve scale if some sensible value was recorded
            ylim = [max(ylim), min(ylim)]  # Make sure y-axis is inverted
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            self._ax.set_autoscale_on(False)

    def run_async(self, runnable, result_handler):
        """Creates a callable that runs the given runnable on a worker thread."""
        class MyTask(Task):
            def compute(self):
                return runnable()

            def on_finished(self, result: Any):
                result_handler(result)

        def internal():
            self._window.get_scheduler().add_task(MyTask())
        return internal

    def draw_view(self):
        """Draws the view."""
        raise NotImplementedError()

    def refresh_data(self):
        """Redraws the view."""
        self.draw_view()

    def refresh_all(self):
        """Redraws the view after loading the images."""
        self._window.setup_menu(self.get_extra_menu_options(), show_plugins=self._get_must_show_plugin_menus())
        self._window.set_window_title(self._get_window_title())
        self.draw_view()

    def update_status(self, text: Union[str, bytes], redraw=True):
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
                self.update_status("Unknown command: " + text)
        except Exception as e:
            dialog.popup_exception(e)

    def _on_command(self, text: str) -> bool:
        return False

    def _on_mouse_click(self, event: MouseEvent):
        pass

    def _on_scroll(self, event: MouseEvent):
        """Called when scrolling. event.button will be "up" or "down"."""
        pass

    def attach(self):
        self._window.setup_menu(self.get_extra_menu_options(), show_plugins=self._get_must_show_plugin_menus())
        self._window.set_window_title(self._get_window_title())
        self._window.register_event_handler("key_press_event", self._on_key_press_raw)
        self._window.register_event_handler("button_press_event", self._on_mouse_click)
        self._window.register_event_handler("data_updated_event", self.refresh_data)
        self._window.register_event_handler("any_updated_event", self.refresh_all)
        self._window.register_event_handler("command_event", self._on_command_raw)
        self._window.register_event_handler("scroll_event", self._on_scroll)

    def detach(self):
        self._window.unregister_event_handlers()

    def _get_window_title(self) -> Optional[str]:
        """Called to query what the window title should be. This will be prefixed with the name of the program."""
        return None

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {}

    def _get_must_show_plugin_menus(self) -> bool:
        """Returns whether the plugin-added menu options must be shown in this visualizer."""
        return False

    def get_default_status(self) -> str:
        """Gets the status normally used when moving between time points or between different visualizers. Use
        update_status to set a special status."""
        return str(self.__doc__)

    def load_image(self, time_point: TimePoint, show_next_time_point: bool) -> Optional[ndarray]:
        """Creates an image suitable for display purposes. IF show_next_time_point is set to True, then then a color
        image will be created with the next image in red, and the current image in green."""
        time_point_images = self._experiment.images.get_image_stack(time_point, self._display_settings.image_channel)
        if time_point_images is None:
            return None
        if show_next_time_point:
            image_shape = time_point_images.shape

            rgb_images = numpy.zeros((image_shape[0], image_shape[1], image_shape[2], 3), dtype='float')
            rgb_images[:,:,:,1] = time_point_images  # Green channel is current image
            try:
                next_time_point = self._experiment.get_next_time_point(time_point)
                next_time_point_images = self._experiment.images.get_image_stack(next_time_point,
                                                                                 self._display_settings.image_channel)

                # Check if we need to translate the next image
                offsets = self._experiment.images.offsets
                relative_offset = offsets.of_time_point(time_point) - offsets.of_time_point(next_time_point)
                if relative_offset.x != 0 or relative_offset.y != 0 or relative_offset.z != 0:
                    original_images = next_time_point_images
                    next_time_point_images = numpy.zeros_like(original_images)
                    cropper.crop_3d(original_images, int(relative_offset.x), int(relative_offset.y),
                                    int(relative_offset.z), output=next_time_point_images)

                rgb_images[:,:,:,0] = next_time_point_images # Red channel is next image
            except ValueError:
                pass  # There is no next time point, ignore
            rgb_images /= rgb_images.max()
            time_point_images = rgb_images
        return time_point_images

    def reconstruct_image(self, time_point: TimePoint, rgb_canvas: ndarray):
        """Draws all positions and shapes to the given canvas. The canvas must be a float array,
        and will be clipped from 0 to 1."""
        offset = self._experiment.images.offsets.of_time_point(time_point)
        colors = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        i = 0
        for position, shape in self._experiment.positions.of_time_point_with_shapes(time_point):
            shape.draw3d_color(position.x - offset.x, position.y - offset.y, position.z - offset.z, 0, rgb_canvas, colors[i % len(colors)])
            i += 1
        rgb_canvas.clip(min=0, max=1, out=rgb_canvas)

    def _get_type_color(self, position: Position) -> Optional[MPLColor]:
        """Gets the color that the given position should be annotated with, based on the type of the position. Usually
        this color is used to decorate the edge of the position marker."""
        position_type = self.get_window().get_gui_experiment().get_marker_by_save_name(
            linking_markers.get_position_type(self._experiment.links, position))
        if position_type is None:
            return None
        return position_type.mpl_color

    @staticmethod
    def get_closest_position(positions: Iterable[Position], x: Optional[int], y: Optional[int], z: Optional[int],
                             time_point: Optional[TimePoint], max_distance: int = 100000):
        """Gets the position closest ot the given position. If z is missing, it is ignored. If x or y are missing,
        None is returned.
        """
        if x is None or y is None or time_point is None:
            return None # Mouse outside figure, so x or y are None
        ignore_z = False
        if z is None:
            z = 0
            ignore_z = True
        resolution = ImageResolution(1, 1, 6, 1)  # Just use a random resolution - it's just for clicking after all
        search_position = Position(x, y, z, time_point=time_point)
        return find_closest_position(positions, around=search_position, ignore_z=ignore_z,
                                     max_distance_um=max_distance, resolution=resolution)

    def get_window(self) -> Window:
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
    __active_visualizer.update_status(__active_visualizer.get_default_status())

