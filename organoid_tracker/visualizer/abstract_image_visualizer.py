from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy
from matplotlib import cm
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from numpy import ndarray
from tifffile import tifffile

from organoid_tracker import core
from organoid_tracker.core import TimePoint, UserError, COLOR_CELL_CURRENT
from organoid_tracker.core.position import Position
from organoid_tracker.core.spline import Spline
from organoid_tracker.core.typing import MPLColor
from organoid_tracker.gui import dialog
from organoid_tracker.gui.dialog import prompt_int
from organoid_tracker.gui.window import Window, DisplaySettings
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.util.mpl_helper import line_infinite
from organoid_tracker.visualizer import Visualizer, activate


class AbstractImageVisualizer(Visualizer):
    """A generic image visualizer."""

    MAX_Z_DISTANCE: int = 3
    DEFAULT_SIZE = (30, 500, 500)

    _time_point_images: Optional[ndarray] = None

    # The color map should typically not be transferred when switching to another viewer, so it is not part of the
    # display_settings property
    _color_map: Colormap = cm.get_cmap("gray")

    def __init__(self, window: Window):
        super().__init__(window)

        self._clamp_time_point()
        self._clamp_z()
        self._load_time_point(self._time_point)

    def _load_time_point(self, time_point: TimePoint):
        """Loads the images and other data of the time point."""
        if self._experiment.first_time_point_number() is None or \
                time_point.time_point_number() < self._experiment.first_time_point_number() or \
                time_point.time_point_number() > self._experiment.last_time_point_number():
            # Experiment has no data (for this time point)
            self._display_settings.time_point = time_point
            self._time_point_images = None
            return

        self._clamp_channel()
        time_point_images = None
        if self._display_settings.show_images:
            time_point_images = self.load_image(time_point, self._display_settings.show_next_time_point)
        if self._display_settings.show_reconstruction:
            if time_point_images is not None:
                # Create background based on time point images
                image_shape = time_point_images.shape[0:3] + (3,)
                rgb_image = numpy.zeros(image_shape, dtype=numpy.float32)

                # Convert to colored float
                if len(time_point_images.shape) == 4:
                    rgb_image[...] = time_point_images
                else:
                    rgb_image[:, :, :, 0] = time_point_images
                    rgb_image[:, :, :, 1] = time_point_images
                    rgb_image[:, :, :, 2] = time_point_images
                rgb_image /= (rgb_image.max() * 2)
                rgb_image.clip(0, 0.25, out=rgb_image)
            else:
                # Create empty background
                image_shape = self._guess_image_size(time_point) + (3,)
                rgb_image = numpy.zeros(image_shape, dtype=numpy.float32)

            # Create reconstruction
            self.reconstruct_image(time_point, rgb_image)
            time_point_images = rgb_image

        self._display_settings.time_point = time_point
        self._time_point_images = time_point_images
        self._clamp_z()

    def _guess_image_size(self, time_point) -> Tuple[int, int, int]:
        images_for_size = self._time_point_images
        if images_for_size is None:
            images_for_size = self.load_image(time_point, show_next_time_point=False)
        size = images_for_size.shape[0:3] if images_for_size is not None else self.DEFAULT_SIZE
        return size

    def refresh_data(self):
        if self._display_settings.show_reconstruction:
            self._load_time_point(self._time_point)  # Reload image, as image is a reconstruction of the data
        super().refresh_data()

    def refresh_all(self):
        self._load_time_point(self._time_point)  # Reload image
        super().refresh_all()

    def draw_view(self):
        self._clear_axis()
        self._ax.set_facecolor((0.2, 0.2, 0.2))
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

    def _draw_line(self, start: Position, end: Position, color: MPLColor = COLOR_CELL_CURRENT):
        direction = end - start
        if direction.x < 0.2 and direction.y < 0.2:
            # Difficult to draw lines in the Z direction, so just draw a marker
            self._ax.scatter(start.x, end.y, marker='o', facecolor=color,
                             edgecolors="black", s=17 ** 2, linewidths=2)
            return

        line_width = 2
        if direction.z == 0:
            if round(start.z) == self._z:
                line_width = 5
        else:
            # Just add a marker
            dz = self._z - start.z
            multiplier = dz / direction.z
            marker_pos = start + direction * multiplier
            self._ax.scatter(marker_pos.x, marker_pos.y, marker='o', facecolor=color,
                             edgecolors="black", s=17 ** 2, linewidths=2)

        line_infinite(self._ax, start.x, start.y, end.x, end.y, linewidth=line_width, color=color)

    def _get_figure_title(self) -> str:
        return "Time point " + str(self._time_point.time_point_number()) + "    (z=" + str(self._z) + ")"

    def _draw_extra(self):
        pass  # Subclasses can override this

    def _must_show_other_time_points(self) -> bool:
        return True

    def _must_draw_positions_of_previous_time_point(self) -> bool:
        """Returns whether the positions of the previous time point are drawn based on the current display settings."""
        # Only draw positions of previous time point if there is linking data, or if we're forced to.
        can_show_other_time_points = self._must_show_other_time_points() and self._experiment.links.has_links()
        return not self._display_settings.show_next_time_point and can_show_other_time_points

    def _must_draw_positions_of_next_time_point(self) -> bool:
        """Returns whether the positions of the next time point are drawn based on the current display settings."""
        # Only draw positions of next time point if there is linking data, or if we're forced to.
        can_show_other_time_points = self._must_show_other_time_points() and self._experiment.links.has_links()
        return self._display_settings.show_next_time_point or can_show_other_time_points

    def _draw_positions(self):
        """Draws positions and links. Returns the amount of non-equal links in the image"""
        if not self._display_settings.show_positions:
            return

        # Next time point
        if self._must_draw_positions_of_next_time_point():

            try:
                self._draw_positions_of_time_point(self._experiment.get_next_time_point(self._time_point),
                                                   color=core.COLOR_CELL_NEXT)
            except ValueError:
                pass  # There is no next time point, ignore

        # Previous time point
        if self._must_draw_positions_of_previous_time_point():
            try:
                self._draw_positions_of_time_point(
                    self._experiment.get_previous_time_point(self._time_point), color=core.COLOR_CELL_PREVIOUS)
            except ValueError:
                pass  # There is no previous time point, ignore

        # Current time point
        self._draw_positions_of_time_point(self._time_point)

    def _draw_positions_of_time_point(self, time_point: TimePoint, color: str = core.COLOR_CELL_CURRENT):
        position_data = self._experiment.position_data
        dt = time_point.time_point_number() - self._time_point.time_point_number()
        show_errors = self._display_settings.show_errors

        positions_x_list, positions_y_list, positions_edge_colors, positions_edge_widths, positions_marker_sizes =\
            list(), list(), list(), list(), list()
        crosses_x_list, crosses_y_list = list(), list()

        min_z, max_z = self._z - self.MAX_Z_DISTANCE, self._z + self.MAX_Z_DISTANCE
        for position in self._experiment.positions.of_time_point_and_z(time_point, min_z, max_z):
            dz = self._z - round(position.z)

            # Draw the position, making it selectable
            if not self._on_position_draw(position, color, dz, dt):
                continue

            # Add error marker
            if show_errors and linking_markers.get_error_marker(position_data, position) is not None:
                crosses_x_list.append(position.x)
                crosses_y_list.append(position.y)

            # Add marker
            position_type = self.get_window().get_gui_experiment().get_marker_by_save_name(
                linking_markers.get_position_type(position_data, position))
            edge_color = (0, 0, 0) if position_type is None else position_type.mpl_color
            edge_width = 1 if position_type is None else 3

            positions_x_list.append(position.x)
            positions_y_list.append(position.y)
            positions_edge_colors.append(edge_color)
            positions_edge_widths.append(edge_width)
            dz_penalty = 0 if dz == 0 else abs(dz) + 1
            positions_marker_sizes.append(max(1, 7 - dz_penalty - abs(dt) + edge_width) ** 2)

        self._ax.scatter(crosses_x_list, crosses_y_list, marker='X', facecolor='black', edgecolors="white",
                         s=17**2, linewidths=2)
        marker = "s" if dt == 0 else "o"
        self._ax.scatter(positions_x_list, positions_y_list, s=positions_marker_sizes, facecolor=color,
                         edgecolors=positions_edge_colors, linewidths=positions_edge_widths, marker=marker)

    def _on_position_draw(self, position: Position, color: str, dz: int, dt: int) -> bool:
        """Called whenever a position is being drawn. Return False to prevent drawing of this position."""
        return True

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
        if not self._display_settings.show_positions:
            return

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
        min_z = self._z - self.MAX_Z_DISTANCE
        max_z = self._z + self.MAX_Z_DISTANCE

        # Find all drawn positions
        selectable_positions = set(self._experiment.positions.of_time_point_and_z(self._time_point, min_z, max_z))
        if self._must_draw_positions_of_previous_time_point():
            try:
                previous_time_point = self._experiment.get_previous_time_point(self._time_point)
            except ValueError:
                pass
            else:
                for position in self._experiment.positions.of_time_point_and_z(previous_time_point, min_z, max_z):
                    selectable_positions.add(position)
        if self._must_draw_positions_of_next_time_point():
            try:
                next_time_point = self._experiment.get_next_time_point(self._time_point)
            except ValueError:
                pass
            else:
                for position in self._experiment.positions.of_time_point_and_z(next_time_point, min_z, max_z):
                    selectable_positions.add(position)

        # Find nearest position
        return self.get_closest_position(selectable_positions, x, y, None, self._time_point, max_distance=5)

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
            "View//Toggle-Toggle showing position markers [P]": self._toggle_showing_position_markers,
            "View//Toggle-Toggle showing error markers": self._toggle_showing_error_markers,
            "View//Image-View image slices [\]": self._show_slices,
            "Navigate//Layer-Above layer [Up]": lambda: self._move_in_z(1),
            "Navigate//Layer-Below layer [Down]": lambda: self._move_in_z(-1),
            "Navigate//Channel-Next channel [.]": lambda: self._move_in_channel(1),
            "Navigate//Channel-Previous channel [,]": lambda: self._move_in_channel(-1),
            "Navigate//Time-Next time point [Right]": lambda: self._move_in_time(1),
            "Navigate//Time-Previous time point [Left]": lambda: self._move_in_time(-1),
            "Navigate//Time-First time point [F]": self._move_to_first_time_point,
            "Navigate//Time-Other time point... (/t*)": time_point_prompt
        }

    def _on_scroll(self, event: MouseEvent):
        """Move in z."""
        if event.button == 'up':
            self._move_in_z(1)
        else:
            self._move_in_z(-1)

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
        if len(command) > 0 and command[0] == "z":
            z_str = command[1:]
            try:
                new_z = int(z_str.strip())
                if self._move_to_z(new_z):
                    self.update_status(f"Moved to z {self._display_settings.z}!")
                else:
                    self.update_status(f"Z layer {new_z} does not exist.")
            except ValueError:
                self.update_status("Cannot read number: " + z_str)
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
            self._display_settings.show_images = True
            self._display_settings.show_positions = True
            self._display_settings.show_splines = True
            self._display_settings.show_reconstruction = False
            self._display_settings.show_next_time_point = False
            self.refresh_all()
            self.update_status("You're already in the home screen. Reset most display settings.")
            return True
        return False

    def _export_3d_image(self):
        if self._time_point_images is None:
            raise core.UserError("No images loaded", "Saving images failed: there are no images loaded")
        file = dialog.prompt_save_file("Save 3D file as...", [("TIF file", "*.tif")])
        if file is None:
            return

        images: ndarray = cv2.convertScaleAbs(self._time_point_images, alpha=256 / self._time_point_images.max(), beta=0)
        image_shape = self._time_point_images.shape

        if len(image_shape) == 3 and isinstance(self._color_map, Colormap):
            # Convert grayscale image to colored using the stored color map
            flat_image = images.ravel()
            images = self._color_map(flat_image, bytes=True)[:, 0:3]
            new_shape = (image_shape[0], image_shape[1], image_shape[2], 3)
            images = images.reshape(new_shape)
        else:
            # Color images can be kept as is - they were already rescaled by the convertScaleAbs function, and no
            # other transformations are necessary
            pass

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

    def _toggle_showing_position_markers(self):
        self._display_settings.show_positions = not self._display_settings.show_positions
        self.draw_view()

    def _toggle_showing_error_markers(self):
        self._display_settings.show_errors = not self._display_settings.show_errors
        self.draw_view()

    def _show_slices(self):
        from organoid_tracker.visualizer.image_slice_visualizer import ImageSliceViewer
        activate(ImageSliceViewer(self._window, self.__class__))

    def _move_in_z(self, dz: int) -> bool:
        return self._move_to_z(self._display_settings.z + dz)

    def _move_to_z(self, new_z: int) -> bool:
        """Moves to another z and redraws. Returns false and does nothing else if the given z does not exist."""
        old_z = self._display_settings.z
        self._display_settings.z = new_z
        self._clamp_z()
        if self._display_settings.z != new_z:
            # Failed, out of range
            self._display_settings.z = old_z
            return False

        if self._display_settings.z != old_z:
            self.draw_view()
        return True

    def _clamp_time_point(self):
        time_point_number = self._time_point.time_point_number()
        min_time_point_number = self._experiment.first_time_point_number()
        max_time_point_number = self._experiment.last_time_point_number()
        if min_time_point_number is not None and time_point_number < min_time_point_number:
            self._display_settings.time_point = TimePoint(min_time_point_number)
        elif max_time_point_number is not None and time_point_number > max_time_point_number:
            self._display_settings.time_point = TimePoint(max_time_point_number)

    def _clamp_z(self):
        """Makes sure a valid z pos is selected. Changes the z if not."""
        image_z_offset = int(self._experiment.images.offsets.of_time_point(self._time_point).z)
        if self._display_settings.z < image_z_offset:
            self._display_settings.z = image_z_offset
        if self._time_point_images is not None\
                and self._display_settings.z >= len(self._time_point_images) + image_z_offset:
            self._display_settings.z = len(self._time_point_images) + image_z_offset - 1

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

    def _move_to_first_time_point(self) -> bool:
        first_Time_point_number = self._experiment.first_time_point_number()
        if first_Time_point_number is None:
            self.update_status("Cannot move to the first time point. There are no time points loaded.")
            return False
        if first_Time_point_number == self._time_point.time_point_number():
            self.update_status("You are already in the first time point.")
            return False
        self._move_to_time(first_Time_point_number)

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
            self._display_settings.z = round(position.z)
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
