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
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import ndarray

from organoid_tracker import core
from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.imaging import cropper
from organoid_tracker.linking.nearby_position_finder import find_closest_position
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.util import mpl_helper


class Visualizer:
    """A complete application for visualization of an experiment"""
    _window: Window
    _fig: Figure
    _ax: Axes
    _axes: List[Axes]
    _display_settings: DisplaySettings

    # Used to detect whether a mouse moved while pressed, so that dragging can be detected
    _mouse_press_x: Optional[float] = None
    _mouse_press_y: Optional[float] = None

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

    def _get_color_map(self) -> Colormap:
        """Returns the color map to use for the images. Based on the currently displayed channel and the stored
        colormap for that channel."""
        return self._experiment.images.get_channel_description(self._display_settings.image_channel).colormap

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

    def _on_mouse_move_raw(self, event: MouseEvent):
        """Calls _on_mouse_move, but catches all exceptions."""
        try:
            self._on_mouse_move(event)
        except Exception as e:
            dialog.popup_exception(e)

    def _on_mouse_move(self, event: MouseEvent):
        """Called when the mouse is moved."""
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

    def _on_mouse_single_click(self, event: MouseEvent):
        """Called at mouse release, if no drag or double-click was detected."""
        pass

    def _on_mouse_double_click(self, event: MouseEvent):
        """Called at mouse double click."""
        pass

    def _on_mouse_press_raw(self, event: MouseEvent):
        if event.dblclick:
            # Perform double-click event
            self._on_mouse_double_click(event)
            self._mouse_press_x = None
            self._mouse_press_y = None
        else:
            # Record mouse position for release event later on
            self._mouse_press_x = event.x
            self._mouse_press_y = event.y

    def _on_mouse_release_raw(self, event: MouseEvent):
        """Called when the mouse is released. If the mouse was not moved, then _on_mouse_click is called."""
        if self._mouse_press_x is None or self._mouse_press_y is None:
            return  # No mouse press event was recorded

        distance_squared = (event.x - self._mouse_press_x) ** 2 + (event.y - self._mouse_press_y) ** 2
        if distance_squared < 3 and not event.dblclick:
            self._on_mouse_single_click(event)
        self._mouse_press_x = None
        self._mouse_press_y = None


    def _on_scroll(self, event: MouseEvent):
        """Called when scrolling. event.button will be "up" or "down"."""
        pass

    def _on_program_close(self):
        """Called when the program is being closed. The user will already have confirmed the closing, so you cannot
        block the closing at this point."""
        pass

    def attach(self):
        """Attaches all event handlers."""
        self._window.register_event_handler("key_press_event", self._on_key_press_raw)
        self._window.register_event_handler("button_press_event", self._on_mouse_press_raw)
        self._window.register_event_handler("button_release_event", self._on_mouse_release_raw)
        self._window.register_event_handler("motion_notify_event", self._on_mouse_move)
        self._window.register_event_handler("data_updated_event", self.refresh_data)
        self._window.register_event_handler("any_updated_event", self.refresh_all)
        self._window.register_event_handler("command_event", self._on_command_raw)
        self._window.register_event_handler("scroll_event", self._on_scroll)
        self._window.register_event_handler("program_close_event", self._on_program_close)

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

    def _return_2d_image(self, time_point: TimePoint, z: int, channel: ImageChannel, show_next_time_point: bool) -> Optional[ndarray]:
        """Returns the 2D image slice for the given time point, Z and channel. Ignores any current display settings,
        and only uses the settings provided as method arguments. If show_next_time_point is True, then the next time
        point is included in the image as well in red+blue, while the current time point is green."""
        time_point_image = self._experiment.images.get_image_slice_2d(time_point, channel, z)
        if time_point_image is None:
            return None
        if show_next_time_point:
            image_shape = time_point_image.shape

            rgb_images = numpy.zeros((image_shape[0], image_shape[1], 3), dtype=numpy.float32)
            rgb_images[:, :, 1] = time_point_image  # Green channel is current image
            rgb_images[:, :, 1] /= max(1, rgb_images[:, :, 1].max())  # Normalize the green channel
            next_time_point_image = self._experiment.images.get_image_slice_2d(time_point + 1, channel, z)
            if next_time_point_image is None:
                next_time_point_image = numpy.zeros_like(time_point_image)

            # Check if we need to translate the next image
            offsets = self._experiment.images.offsets
            relative_offset = offsets.of_time_point(time_point) - offsets.of_time_point(time_point + 1)
            if relative_offset.x != 0 or relative_offset.y != 0 or relative_offset.z != 0:
                original_images = next_time_point_image
                next_time_point_image = numpy.zeros_like(original_images)
                cropper.crop_2d(original_images, int(relative_offset.x), int(relative_offset.y),
                                output=next_time_point_image)
            rgb_images[:, :, 0] = next_time_point_image  # Red channel is next image
            rgb_images[:, :, 0] /= max(1, rgb_images[:, :, 0].max())  # Normalize the red channel
            rgb_images[:, :, 2] = rgb_images[:, :, 0]  # Blue channel is the same as red channel, to create purple

            time_point_image = rgb_images
        return time_point_image

    def _return_3d_image(self, time_point: TimePoint, channel: ImageChannel, show_next_time_point: bool) -> Optional[ndarray]:
        """Returns the full image stack for the given time pointand channel. Ignores any current display settings,
        and only uses the settings provided as method arguments. If show_next_time_point is True, then the next time
        point is included in the image as well in red+blue, while the current time point is green."""
        time_point_image = self._experiment.images.get_image_stack(time_point, channel)
        if time_point_image is None:
            return None
        if show_next_time_point:
            image_shape = time_point_image.shape

            rgb_images = numpy.zeros((image_shape[0], image_shape[1], image_shape[2], 3), dtype=numpy.float32)
            rgb_images[:, :, :, 1] = time_point_image  # Green channel is current image
            rgb_images[:, :, :, 1] /= max(1, rgb_images[:, :, :, 1].max())  # Normalize the green channel

            next_time_point_image = self._experiment.images.get_image_stack(time_point + 1, channel)
            if next_time_point_image is None:
                next_time_point_image = numpy.zeros_like(time_point_image)

            # Check if we need to translate the next image
            offsets = self._experiment.images.offsets
            relative_offset = offsets.of_time_point(time_point) - offsets.of_time_point(time_point + 1)
            if relative_offset.x != 0 or relative_offset.y != 0 or relative_offset.z != 0:
                original_images = next_time_point_image
                next_time_point_image = numpy.zeros_like(original_images)
                cropper.crop_3d(original_images, int(relative_offset.x), int(relative_offset.y), int(relative_offset.z),
                                output=next_time_point_image)
            rgb_images[:, :, :, 0] = next_time_point_image  # Red channel is next image
            rgb_images[:, :, :, 0] /= max(1, rgb_images[:, :, :, 0].max())  # Normalize the red channel
            rgb_images[:, :, :, 2] = rgb_images[:, :, :, 0]  # Blue channel is the same as red channel, to create purple

            time_point_image = rgb_images
        return time_point_image

    def should_show_image_reconstruction(self) -> bool:
        """If overridden, the reconstruct_image and reconstruct_image_3d can be called."""
        return False

    def reconstruct_image(self, time_point: TimePoint, z: int, rgb_canvas_2d: ndarray):
        """Draws all positions and shapes to the given canvas. The canvas must be a float array,
        and will be clipped from 0 to 1.

        Only called if self._should_show_image_reconstruction returns True.
        """
        pass

    def reconstruct_image_3d(self, time_point: TimePoint, rgb_canvas_3d: ndarray):
        """Draws all positions and shapes to the given canvas. The canvas must be a float array,
        and will be clipped from 0 to 1."""
        pass

    def _get_type_color(self, position: Position) -> Optional[MPLColor]:
        """Gets the color that the given position should be annotated with, based on the type of the position. Usually
        this color is used to decorate the edge of the position marker."""
        position_type = self.get_window().registry.get_marker_by_save_name(
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

