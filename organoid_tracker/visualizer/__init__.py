"""
OrganoidTracker is centered around a Matplotlib window, and that window is controlled here.

Each screen in OrganoidTracker inherits from the Visualizer class, which just shows an empty Matplotlib plot. The
subclasses show more interesting behaviour, like
:class:`~organoid_tracker.visualizer.abstract_image_visualizer.AbstractImageVisualizer` which shows an image from your
time-lapse movie, with the positions and links drawn on top.

If you want to create your own screeen, I would recommend that you create a subclass from ExitableImageVisualzer:

>>> from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer
>>> class YourVisualizer(ExitableImageVisualizer):
>>>     def get_extra_menu_options(self) -> Dict[str, Any]:
>>>         return { "My menu//Some option": self._clicked_my_menu_option }
>>>
>>>     def _clicked_my_menu_option(self):
>>>         self.update_status("Clicked on my menu option!")
>>>
>>>     def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
>>>         if dt == 0 and dz == 0:
>>>             self._ax.text(position.x, position.y, "Some text")  # Draws some text at the positions
>>>         return True  # Change this to False to no longer draw the original marker
"""
from typing import Iterable, Optional, Union, Dict, Any, List, Callable

import numpy
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, Axes
from numpy import ndarray

from organoid_tracker import core
from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.imaging import cropper
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.linking.nearby_position_finder import find_closest_position
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.util import mpl_helper


class Visualizer:
    """A complete application for visualization of an experiment"""
    _window: Window
    _fig: Figure
    _ax: Axes
    _axes: List[Axes]
    _display_settings: DisplaySettings

    def __init__(self, window: Window):
        self._window = window
        self._fig = window.get_figure()

        # Replace axes in figure while keeping old zoom
        old_axes_limits = mpl_helper.store_axes_limits(self._fig.axes[0]) if len(self._fig.axes) > 0 else None
        subplots_config = self._get_subplots_config()
        self._fig.clear(keep_observers=True)
        self._fig.subplots(**subplots_config)
        self._axes = self._fig.axes
        self._ax = self._axes[0]
        mpl_helper.restore_axes_limits(self._ax, old_axes_limits)

    def _get_subplots_config(self) -> Dict[str, Any]:
        """Gets the configuration, passed to figure.subplots. Make sure to at least specify nrows and ncols."""
        return {
            "nrows": 1,
            "ncols": 1
        }

    @property
    def _experiment(self) -> Experiment:
        """By making this a dynamic property (instead of just using self._experiment = window.get_experiment()), its
        value is always up-to-date, even if window.set_experiment(...) is called."""
        try:
            return self._window.get_experiment()
        except UserError:
            # Just use an empty experiment
            return Experiment()

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
        for ax in self._axes:
            for image in ax.images:
                colorbar = image.colorbar
                if colorbar is not None:
                    colorbar.remove_connection()
            for text in self._fig.texts:
                text.remove_connection()

            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            ax.clear()
            if xlim[1] - xlim[0] > 2:
                # Only preserve scale if some sensible value was recorded
                ylim = [max(ylim), min(ylim)]  # Make sure y-axis is inverted
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_autoscale_on(False)

    def run_async(self, runnable: Callable[[], Any], result_handler: Callable[[Any], None]):
        """Creates a callable that runs the given runnable on a worker thread."""
        class MyTask(Task):
            def compute(self):
                return runnable()

            def on_finished(self, result: Any):
                result_handler(result)

        self._window.get_scheduler().add_task(MyTask())

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
        """Attaches all event handlers."""
        self._window.register_event_handler("key_press_event", self._on_key_press_raw)
        self._window.register_event_handler("button_press_event", self._on_mouse_click)
        self._window.register_event_handler("data_updated_event", self.refresh_data)
        self._window.register_event_handler("any_updated_event", self.refresh_all)
        self._window.register_event_handler("command_event", self._on_command_raw)
        self._window.register_event_handler("scroll_event", self._on_scroll)

    def detach(self):
        """Detaches the event handlers."""
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

    def load_image(self, time_point: TimePoint, z: int, show_next_time_point: bool) -> Optional[ndarray]:
        """Creates an image suitable for display purposes. IF show_next_time_point is set to True, then then a color
        image will be created with the next image in red, and the current image in green."""
        channel = self._display_settings.image_channel
        time_point_image = self._experiment.images.get_image_slice_2d(time_point, channel, z)
        if time_point_image is None:
            return None
        if show_next_time_point:
            image_shape = time_point_image.shape

            rgb_images = numpy.zeros((image_shape[0], image_shape[1], 3), dtype='float')
            rgb_images[:,:,1] = time_point_image  # Green channel is current image
            try:
                next_time_point = self._experiment.get_next_time_point(time_point)
                next_time_point_image = self._experiment.images.get_image_slice_2d(next_time_point, channel, z)
                if next_time_point_image is None:
                    next_time_point_image = numpy.zeros_like(time_point_image)

                # Check if we need to translate the next image
                offsets = self._experiment.images.offsets
                relative_offset = offsets.of_time_point(time_point) - offsets.of_time_point(next_time_point)
                if relative_offset.x != 0 or relative_offset.y != 0 or relative_offset.z != 0:
                    original_images = next_time_point_image
                    next_time_point_image = numpy.zeros_like(original_images)
                    cropper.crop_2d(original_images, int(relative_offset.x), int(relative_offset.y),
                                    output=next_time_point_image)
                rgb_images[:,:,0] = next_time_point_image  # Red channel is next image
            except ValueError:
                pass  # There is no next time point, ignore
            rgb_images /= rgb_images.max()
            time_point_image = rgb_images
        return time_point_image

    def reconstruct_image(self, time_point: TimePoint, z: int, rgb_canvas_2d: ndarray):
        """Draws all positions and shapes to the given canvas. The canvas must be a float array,
        and will be clipped from 0 to 1."""
        offset = self._experiment.images.offsets.of_time_point(time_point)
        position_data = self._experiment.position_data
        colors = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        i = 0

        for position in self._experiment.positions.of_time_point(time_point):
            shape = linking_markers.get_shape(position_data, position)
            shape.draw2d_image(position.x - offset.x, position.y - offset.y, z - int(position.z - offset.z),
                               0, rgb_canvas_2d, colors[i % len(colors)])
            i += 1
        rgb_canvas_2d.clip(min=0, max=1, out=rgb_canvas_2d)

    def reconstruct_image_3d(self, time_point: TimePoint, rgb_canvas_3d: ndarray):
        """Draws all positions and shapes to the given canvas. The canvas must be a float array,
        and will be clipped from 0 to 1."""
        offset = self._experiment.images.offsets.of_time_point(time_point)
        position_data = self._experiment.position_data
        colors = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        i = 0

        for position in self._experiment.positions.of_time_point(time_point):
            shape = linking_markers.get_shape(position_data, position)
            shape.draw3d_color(position.x - offset.x, position.y - offset.y, position.z - offset.z,
                               0, rgb_canvas_3d, colors[i % len(colors)])
            i += 1
        rgb_canvas_3d.clip(min=0, max=1, out=rgb_canvas_3d)

    def _get_type_color(self, position: Position) -> Optional[MPLColor]:
        """Gets the color that the given position should be annotated with, based on the type of the position. Usually
        this color is used to decorate the edge of the position marker."""
        position_type = self.get_window().get_gui_experiment().get_marker_by_save_name(
            position_markers.get_position_type(self._experiment.position_data, position))
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
    __active_visualizer.refresh_all()
    __active_visualizer.update_status(__active_visualizer.get_default_status())

