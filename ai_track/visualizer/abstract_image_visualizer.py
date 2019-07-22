from typing import List, Union, Optional, Dict, Any

import cv2
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, Normalize
from numpy.core.multiarray import ndarray
from tifffile import tifffile

from ai_track import core
from ai_track.core import TimePoint, UserError
from ai_track.core.position import Position
from ai_track.core.spline import Spline
from ai_track.core.typing import MPLColor
from ai_track.gui import dialog
from ai_track.gui.dialog import prompt_int, popup_error
from ai_track.gui.window import Window
from ai_track.linking_analysis import linking_markers
from ai_track.visualizer import Visualizer, DisplaySettings


class AbstractImageVisualizer(Visualizer):
    """A generic image visualizer."""

    MAX_Z_DISTANCE: int = 3
    DEFAULT_SIZE = (30, 500, 500)

    _time_point: TimePoint = None
    _time_point_images: Optional[ndarray] = None
    _z: int
    __positions_near_visible_layer: List[Position]

    # The color map should typically not be transferred when switching to another viewer, so it is not part of the
    # display_settings property
    _color_map: Colormap = cm.get_cmap("gray")

    def __init__(self, window: Window, *, time_point: Optional[TimePoint] = None, z: int = 14,
                 display_settings: DisplaySettings = None):
        super().__init__(window, display_settings=display_settings)

        if time_point is None:
            time_point = TimePoint(window.get_experiment().first_time_point_number())
        self._z = int(z)
        self._load_time_point(time_point)
        self.__positions_near_visible_layer = []

    def _load_time_point(self, time_point: TimePoint):
        """Loads the images and other data of the time point."""
        if self._experiment.first_time_point_number() is None or \
                time_point.time_point_number() < self._experiment.first_time_point_number() or \
                time_point.time_point_number() > self._experiment.last_time_point_number():
            # Experiment has no data (for this time point)
            self._time_point = time_point
            self._time_point_images = None
            return

        self._clamp_channel()
        if self._display_settings.show_images:
            if self._display_settings.show_reconstruction:
                time_point_images = self.reconstruct_image(time_point, self._guess_image_size(time_point))
            else:
                time_point_images = self.load_image(time_point, self._display_settings.show_next_time_point)
        else:
            time_point_images = None

        self._time_point = time_point
        self._time_point_images = time_point_images
        self._clamp_z()

    def _guess_image_size(self, time_point):
        images_for_size = self._time_point_images
        if images_for_size is None:
            images_for_size = self.load_image(time_point, show_next_time_point=False)
        size = images_for_size.shape if images_for_size is not None else self.DEFAULT_SIZE
        return size

    def refresh_all(self):
        self._load_time_point(self._time_point)  # Reload image
        super().refresh_all()

    def draw_view(self):
        self._clear_axis()
        self._ax.set_facecolor((0.2, 0.2, 0.2))
        self.__positions_near_visible_layer.clear()
        self._draw_image()
        self._draw_positions()
        self._draw_links()
        self._draw_connections()
        self._draw_data_axes()
        self._draw_extra()
        self._window.set_figure_title(self._get_figure_title())

        self._fig.canvas.draw()

    def _draw_image(self):
        if self._time_point_images is not None:
            offset = self._experiment.images.offsets.of_time_point(self._time_point)
            image_z = int(self._z - offset.z)
            image = self._time_point_images[image_z]
            extent = (offset.x, offset.x + image.shape[1], offset.y + image.shape[0], offset.y)
            self._ax.imshow(image, cmap=self._color_map, extent=extent)

    def _draw_selection(self, position: Position, color: MPLColor):
        """Draws a marker for the given position that indicates that the position is selected. Subclasses can call this
        method to show a position selection.

        Note: this method will draw the selection marker even if the given position is in another time point, or even on
        a completely different z layer. So only call this method if you want to have a marker visible."""
        dz = self._z - round(position.z)
        time_point_number = position.time_point_number()
        dt = 0 if time_point_number is None else self._time_point.time_point_number() - time_point_number
        self._ax.plot(position.x, position.y, 'o', markersize=25, color=(0, 0, 0, 0), markeredgecolor=color, markeredgewidth=5)
        self._experiment.positions.get_shape(position).draw2d(
            position.x, position.y, dz, dt, self._ax, color, "black")

    def _draw_annotation(self, position: Position, text: str, *, text_color: MPLColor = "black",
                         background_color: MPLColor = (1, 1, 1, 0.8)):
        """Draws an annotation with text for the given position."""
        font_size = max(8, 12 - abs(position.z - self._z))
        self._ax.annotate(text, (position.x, position.y), fontsize=font_size,
                          fontweight="bold", color=text_color, backgroundcolor=background_color,
                          horizontalalignment='center', verticalalignment='center')

    def _get_figure_title(self) -> str:
        return "Time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"

    def _must_show_other_time_points(self) -> bool:
        return True

    def _draw_extra(self):
        pass  # Subclasses can override this

    def _draw_positions(self):
        """Draws positions and links. Returns the amount of non-equal links in the image"""

        # Next time point
        can_show_other_time_points = self._must_show_other_time_points() and self._experiment.links.has_links()
        if self._display_settings.show_next_time_point or can_show_other_time_points:
            # Only draw positions of next/previous time point if there is linking data, or if we're forced to
            try:
                self._draw_positions_of_time_point(self._experiment.get_next_time_point(self._time_point),
                                                   color=core.COLOR_CELL_NEXT)
            except ValueError:
                pass  # There is no next time point, ignore

        # Previous time point
        if not self._display_settings.show_next_time_point and can_show_other_time_points:
            try:
                self._draw_positions_of_time_point(
                    self._experiment.get_previous_time_point(self._time_point), color=core.COLOR_CELL_PREVIOUS)
            except ValueError:
                pass  # There is no previous time point, ignore

        # Current time point
        self._draw_positions_of_time_point(self._time_point)

    def _draw_positions_of_time_point(self, time_point: TimePoint, color: str = core.COLOR_CELL_CURRENT):
        links = self._experiment.links
        dt = time_point.time_point_number() - self._time_point.time_point_number()

        circles_x_list, circles_y_list, circles_edge_colors, circles_marker_sizes = list(), list(), list(), list()
        crosses_x_list, crosses_y_list = list(), list()
        squares_x_list, squares_y_list, squares_edge_colors = list(), list(), list()
        square_marker_size = max(1, 7 - abs(dt)) ** 2

        for position in self._experiment.positions.of_time_point(time_point):
            dz = self._z - round(position.z)
            if abs(dz) > self.MAX_Z_DISTANCE:
                continue

            # Draw the position, making it selectable
            self._on_position_draw(position, color, dz, dt)

            # Add error marker
            if linking_markers.get_error_marker(links, position) is not None:
                crosses_x_list.append(position.x)
                crosses_y_list.append(position.y)

            # Add marker
            position_type = self.get_window().get_gui_experiment().get_marker_by_save_name(
                linking_markers.get_position_type(links, position))
            edge_color = (0, 0, 0) if position_type is None else position_type.mpl_color
            if dz != 0:
                # Draw position as circle
                circles_x_list.append(position.x)
                circles_y_list.append(position.y)
                circles_edge_colors.append(edge_color)
                circles_marker_sizes.append(max(1, 7 - abs(dz) - abs(dt)) ** 2)
            else:
                # Draw position as square
                squares_x_list.append(position.x)
                squares_y_list.append(position.y)
                squares_edge_colors.append(edge_color)

        self._ax.scatter(crosses_x_list, crosses_y_list, marker='X', facecolor='black', edgecolors="white",
                         s=17**2, linewidths=2)
        self._ax.scatter(circles_x_list, circles_y_list, s=circles_marker_sizes, facecolor=color,
                         edgecolors=circles_edge_colors, linewidths=1, marker="o")
        self._ax.scatter(squares_x_list, squares_y_list, s=square_marker_size, facecolor=color,
                         edgecolors=squares_edge_colors, linewidths=1, marker="s")

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int):
        """Called whenever a position is being drawn."""
        # Make position selectable
        self.__positions_near_visible_layer.append(position)

    def _draw_connections(self):
        """Draws all connections. A connection indicates that two positions are not the same, but are related."""
        lines = []
        for position1, position2 in self._experiment.connections.of_time_point(self._time_point):
            min_display_z = min(position1.z, position2.z) - self.MAX_Z_DISTANCE
            max_display_z = max(position1.z, position2.z) + self.MAX_Z_DISTANCE
            if self._z < min_display_z or self._z > max_display_z:
                continue

            line = (position1.x, position1.y), (position2.x, position2.y)
            lines.append(line)
        colors = [core.COLOR_CELL_CURRENT]
        linestyles = ["dotted"]
        linewidths = [2]
        self._ax.add_collection(LineCollection(lines, colors=colors, linestyles=linestyles, linewidths=linewidths))

    def _draw_links(self):
        """Draws all links. A link indicates that one position is the same a another position in another time point."""
        lines = []
        colors = []
        for position1, position2 in self._experiment.links.of_time_point(self._time_point):
            min_display_z = min(position1.z, position2.z) - self.MAX_Z_DISTANCE
            max_display_z = max(position1.z, position2.z) + self.MAX_Z_DISTANCE
            if self._z < min_display_z or self._z > max_display_z:
                continue

            line = (position1.x, position1.y), (position2.x, position2.y)
            lines.append(line)
            color = core.COLOR_CELL_NEXT if position2.time_point_number() > position1.time_point_number()\
                else core.COLOR_CELL_PREVIOUS
            colors.append(color)

        linewidths = [1]
        self._ax.add_collection(LineCollection(lines, colors=colors, linewidths=linewidths))

    def _draw_data_axes(self):
        """Draws the data axis, which is usually the crypt axis."""
        for axis_id, data_axis in self._experiment.splines.of_time_point(self._time_point):
            self._draw_data_axis(data_axis, axis_id, color=core.COLOR_CELL_CURRENT, marker_size_max=10)

    def _draw_data_axis(self, data_axis: Spline, id: int, color: str, marker_size_max: int):
        """Draws a single data axis. Usually, we use this as the crypt axis."""
        if not self._display_settings.show_splines:
            return

        dz = abs(data_axis.get_z() - self._z)
        marker = data_axis.get_direction_marker() if self._experiment.splines.is_axis(id) else "o"
        linewidth = 3 if dz == 0 else 1

        origin_pos = data_axis.from_position_on_axis(0)
        if origin_pos is not None:
            self._ax.plot(origin_pos[0], origin_pos[1], marker="*", markerfacecolor=core.COLOR_CELL_CURRENT,
                          markeredgecolor="black", markersize=max(11, 18 - dz))
        checkpoint = data_axis.get_checkpoint()
        if checkpoint is not None:
            checkpoint_pos = data_axis.from_position_on_axis(checkpoint)
            if checkpoint_pos is not None:
                self._ax.plot(checkpoint_pos[0], checkpoint_pos[1], marker="*", markerfacecolor=core.COLOR_CELL_CURRENT,
                              markeredgecolor="black", markersize=max(11, 18 - dz))

        self._ax.plot(*data_axis.get_interpolation_2d(), color=color, linewidth=linewidth)
        self._ax.plot(*data_axis.get_points_2d(), linewidth=0, marker=marker, markerfacecolor=color,
                      markeredgecolor="black", markersize=max(7, marker_size_max - dz))

    def _get_position_at(self, x: Optional[int], y: Optional[int]) -> Optional[Position]:
        """Wrapper of get_closest_position that makes use of the fact that we can lookup all positions ourselves."""
        return self.get_closest_position(self.__positions_near_visible_layer, x, y, None, self._time_point,
                                         max_distance=5)

    def get_extra_menu_options(self) -> Dict[str, Any]:
        def time_point_prompt():
            min_value = self._experiment.first_time_point_number()
            max_value = self._experiment.last_time_point_number()
            if min_value is None or max_value is None:
                raise UserError("Time point switching", "No data is loaded. Cannot go to another time point.")
            given = prompt_int("Time point", f"Which time point do you want to go to? ({min_value}-{max_value}"
                               + ", inclusive)")
            if given is None:
                return
            if not self._move_to_time(given):
                raise UserError("Out of range", f"Oops, time point {given} is outside the range {min_value}-"
                                f"{max_value}.")
        return {
            **super().get_extra_menu_options(),
            "File//Export-Export 3D image...": self._export_3d_image,
            "View//Toggle-Toggle showing two time points [" + DisplaySettings.KEY_SHOW_NEXT_IMAGE_ON_TOP.upper() + "]":
                self._toggle_showing_next_time_point,
            "View//Toggle-Toggle showing images [" + DisplaySettings.KEY_SHOW_IMAGES.upper() + "]":
                self._toggle_showing_images,
            "View//Toggle-Toggle showing reconstruction [" + DisplaySettings.KEY_SHOW_RECONSTRUCTION.upper() + "]":
                self._toggle_showing_reconstruction,
            "View//Toggle-Toggle showing splines": self._toggle_showing_splines,
            "Navigate//Layer-Above layer [Up]": lambda: self._move_in_z(1),
            "Navigate//Layer-Below layer [Down]": lambda: self._move_in_z(-1),
            "Navigate//Channel-Next channel [.]": lambda: self._move_in_channel(1),
            "Navigate//Channel-Previous channel [,]": lambda: self._move_in_channel(-1),
            "Navigate//Time-Next time point [Right]": lambda: self._move_in_time(1),
            "Navigate//Time-Previous time point [Left]": lambda: self._move_in_time(-1),
            "Navigate//Time-Other time point... (/t*)": time_point_prompt
        }

    def _on_command(self, command: str) -> bool:
        if len(command) > 0 and command[0] == "t":
            time_point_str = command[1:]
            try:
                new_time_point_number = int(time_point_str.strip())
                if not self._move_to_time(new_time_point_number):
                    self.update_status(f"Time point {new_time_point_number} doesn't exist. Available range is "
                                       + str(self._experiment.first_time_point_number()) + " to "
                                       + str(self._experiment.last_time_point_number()) + ", inclusive)")
            except ValueError:
                self.update_status("Cannot read number: " + time_point_str)
            return True
        if command.startswith("goto "):
            split = command.split(" ")
            if len(split) != 5:
                self.update_status("Syntax: /goto <x> <y> <z> <t>")
                return True
            try:
                x, y, z, t = float(split[1]), float(split[2]), float(split[3]), int(split[4])
            except ValueError:
                self.update_status(f"Invalid number in \"{command}\". Syntax: /goto <x> <y> <z> <t>")
            else:
                if not self._move_to_position(Position(x, y, z, time_point_number=t)):
                    self.update_status("Cannot go to point at time point " + str(t))
            return True
        if command == "help":
            self.update_status("/t20: Jump to time point 20 (also works for other time points)"
                               "\n/goto <x> <y> <z> <t>: Directly jump to that point")
            return True
        if command == "exit":
            self.update_status("You're already in the home screen.")
            return True
        return False

    def _export_3d_image(self):
        if self._time_point_images is None:
            raise core.UserError("No images loaded", "Saving images failed: there are no images loaded")
        file = dialog.prompt_save_file("Save 3D file as...", [("TIF file", "*.tif")])
        if file is None:
            return
        flat_image = self._time_point_images.ravel()

        image_shape = self._time_point_images.shape
        if len(image_shape) == 3 and isinstance(self._color_map, Colormap):
            # Convert grayscale image to colored using the stored color map
            images: ndarray = self._color_map(flat_image, bytes=True)[:, 0:3]
            new_shape = (image_shape[0], image_shape[1], image_shape[2], 3)
            images = images.reshape(new_shape)
        else:
            images = cv2.convertScaleAbs(self._time_point_images, alpha=256 / self._time_point_images.max(), beta=0)
        tifffile.imsave(file, images, compress=9)

    def _toggle_showing_next_time_point(self):
        self._display_settings.show_next_time_point = not self._display_settings.show_next_time_point
        self._move_in_time(0)  # Refreshes image

    def _toggle_showing_images(self):
        self._display_settings.show_images = not self._display_settings.show_images
        self._move_in_time(0)  # Refreshes image

    def _toggle_showing_reconstruction(self):
        self._display_settings.show_reconstruction = not self._display_settings.show_reconstruction
        self._move_in_time(0)  # Refreshes image

    def _toggle_showing_splines(self):
        self._display_settings.show_splines = not self._display_settings.show_splines
        self.draw_view()

    def _move_in_z(self, dz: int):
        old_z = self._z
        self._z += dz

        self._clamp_z()

        if self._z != old_z:
            self.draw_view()

    def _clamp_z(self):
        """Makes sure a valid z pos is selected. Changes the z if not."""
        image_z_offset = int(self._experiment.images.offsets.of_time_point(self._time_point).z)
        if self._z < image_z_offset:
            self._z = image_z_offset
        if self._time_point_images is not None and self._z >= len(self._time_point_images) + image_z_offset:
            self._z = len(self._time_point_images) + image_z_offset - 1

    def _clamp_channel(self):
        """Makes sure a valid channel is selected. Changes the channel if not."""
        available_channels = self._experiment.images.image_loader().get_channels()

        # Handle 1 or 0 channels
        if len(available_channels) == 1:
            self._display_settings.image_channel = available_channels[0]
            return
        if len(available_channels) == 0:
            self._display_settings.image_channel = None
            return

        # Handle two or more channels
        try:
            available_channels.index(self._display_settings.image_channel)
        except ValueError:
            # Didn't select a valid channel, switch to first one
            self._display_settings.image_channel = available_channels[0]

    def _move_to_time(self, new_time_point_number: int) -> bool:
        try:
            time_point = self._experiment.get_time_point(new_time_point_number)
        except ValueError:
            return False
        else:
            self._load_time_point(time_point)
            self.draw_view()
            self.update_status("Moved to time point " + str(new_time_point_number) + "!")
            return True

    def _move_to_position(self, position: Position) -> bool:
        try:
            time_point = self._experiment.get_time_point(position.time_point_number())
        except ValueError:
            return False
        else:
            self._load_time_point(time_point)
            self._ax.set_xlim(position.x - 50, position.x + 50)
            self._ax.set_ylim(position.y + 50, position.y - 50)
            self._ax.set_autoscale_on(False)
            self._z = round(position.z)
            self._clamp_z()
            self.draw_view()
            self.update_status(f"Moved to {position}")
            return True

    def _move_in_time(self, dt: int):
        old_time_point_number = self._time_point.time_point_number()
        new_time_point_number = old_time_point_number + dt
        try:
            time_point = self._experiment.get_time_point(new_time_point_number)
        except ValueError:
            pass
        else:
            self._load_time_point(time_point)
            self.draw_view()
            self.update_status(self.get_default_status())

    def _move_in_channel(self, dc: int):
        channels = self._experiment.images.image_loader().get_channels()
        if len(channels) < 2:
            # Nothing to choose, just use the default
            self._display_settings.image_channel = None
            self.update_status("There is only one image channel available, so we cannot switch channels.")
            return

        try:
            old_index = channels.index(self._display_settings.image_channel)
        except ValueError:
            old_index = 0
        new_index = (old_index + dc) % len(channels)
        self._display_settings.image_channel = channels[new_index]

        try:
            self._load_time_point(self._time_point)  # Reload image
            self.draw_view()
            self.update_status(f"Switched to channel {new_index + 1} of {len(channels)}")
        except ValueError:
            pass
