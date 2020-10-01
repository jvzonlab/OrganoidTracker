from enum import Enum
from typing import Optional, Any, Dict

import cv2
import numpy
import tifffile
from numpy import ndarray

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Images
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.util import bits

# How much the pixels are scaled in the output movie. Doesn't affect the points that represent the positions.
_GLOBAL_SCALE = 2
_MAX_DISTANCE_UM_FROM_SLICE = 4


class SliceType(Enum):
    XY = 1  # XY plane, so viewing a single z-coord
    XZ = 2  # XZ plane, so viewing a single y-coord
    YZ = 3  # YZ plane, so viewing a single x-coord

    def get_axis(self) -> str:
        """Gets the axis (lowercase letter) of which only one coordinate is visible. So "z" for an XY slice."""
        if self == SliceType.XY:
            return "z"
        if self == SliceType.XZ:
            return "y"
        return "x"


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export movie//Slice-XY slice...":
            lambda: _generate_slice_movie_prompt(window, SliceType.XY),
        "File//Export-Export movie//Slice-YZ slice...":
            lambda: _generate_slice_movie_prompt(window, SliceType.YZ),
        "File//Export-Export movie//Slice-XZ slice...":
            lambda: _generate_slice_movie_prompt(window, SliceType.XZ)
    }


def _generate_slice_movie_prompt(window: Window, slice_type: SliceType):
    experiment = window.get_experiment()

    # Check if resolution is available
    experiment.images.resolution()

    image_size_zyx = experiment.images.image_loader().get_image_size_zyx()
    if image_size_zyx is None:
        raise UserError("Unknown image size", "Unknown image size. Likely, the image size is not consistent across the"
                                              " time lapse.")
    default_coord = window.display_settings.z if slice_type == SliceType.XY else 200
    coord = dialog.prompt_int(slice_type.get_axis().upper() + " coordinate",
                              f"At which {slice_type.get_axis()} coordinate (in px) do you want to make the slice?",
                              default=default_coord)
    if coord is None:
        return
    output_file = dialog.prompt_save_file("Movie location", [("TIFF file", "*.tif")])
    if output_file is None:
        return

    window.get_scheduler().add_task(_MovieGenerator(experiment, coord, slice_type,
                                                    window.display_settings.image_channel, output_file))


class _MovieGenerator(Task):
    _experiment: Experiment
    _coord: int
    _slice_type: SliceType
    _channel: ImageChannel
    _output_file: str

    def __init__(self, experiment: Experiment, coord: int, slice_type: SliceType, channel: ImageChannel,
                 output_file: str):
        self._experiment = experiment.copy_selected(images=True, positions=True)
        self._coord = coord
        self._slice_type = slice_type
        self._channel = channel
        self._output_file = output_file

    def compute(self) -> Any:
        array = _generate_slice_movie(self._experiment, self._coord, self._slice_type, self._channel)
        tifffile.imwrite(self._output_file, array, compress=9)
        return True

    def on_finished(self, result: Any):
        dialog.popup_message("Movie completed", "Movie has been saved to " + self._output_file + ".")


def _draw_marker(image_rgb: ndarray, image_x: float, image_y: float):
    cv2.circle(image_rgb, (int(image_x), int(image_y)), radius=7, color=(142, 68, 173),  # Magenta
               thickness=-1)


def _generate_slice_movie(experiment: Experiment, coord: int, slice_type: SliceType, channel: ImageChannel) -> ndarray:
    if slice_type == SliceType.XY:
        return _generate_xy_movie(experiment, coord, channel)
    if slice_type == SliceType.XZ:
        return _generate_xz_movie(experiment, coord, channel)
    if slice_type == SliceType.YZ:
        return _generate_yz_movie(experiment, coord, channel)
    raise ValueError("Unknown slice type: " + str(slice_type))


#############
# XY movies #
#############


def _generate_xy_movie(experiment: Experiment, z: int, channel: ImageChannel) -> ndarray:
    """Returns a TYX array, slicing the images."""
    images = experiment.images
    positions = experiment.positions

    # Use time points with positions, except if there are none, then use time points with images
    time_points = list(positions.time_points())
    if len(time_points) == 0:
        time_points = list(images.time_points())

    image_size_zyx = images.image_loader().get_image_size_zyx()

    x_scale = _GLOBAL_SCALE
    y_scale = _GLOBAL_SCALE

    # Shape of array: TYXC
    output_array = numpy.zeros(
        (len(time_points), int(image_size_zyx[1] * y_scale), int(image_size_zyx[2] * x_scale), 3),
        dtype=numpy.uint8)
    temp_array = numpy.zeros(output_array.shape[1:3], dtype=numpy.uint8)  # Just a 2D, uncolored slice
    for i, time_point in enumerate(time_points):
        print(f"Working on time point {time_point.time_point_number()}...")

        # Draw image
        image_slice = images.get_image_slice_2d(time_point, channel, z)
        if image_slice is not None:
            image_slice = bits.ensure_8bit(image_slice)
            cv2.resize(image_slice, dst=temp_array, dsize=(output_array[i].shape[1], output_array[i].shape[0]),
                       fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            output_array[i, :, :, 0] = temp_array
            output_array[i, :, :, 1] = temp_array
            output_array[i, :, :, 2] = temp_array

        # Draw positions
        _draw_xy_positions(output_array[i], experiment, time_point, x_scale, y_scale, z)

    return output_array


def _draw_xy_positions(image_2d_rgb: ndarray, experiment: Experiment, time_point: TimePoint, x_scale: float,
                       y_scale: float, image_z_location: int):
    """Draws the positions on the given 2d RGB image."""
    images = experiment.images
    resolution = images.resolution()
    max_px_from_slice = _MAX_DISTANCE_UM_FROM_SLICE / resolution.pixel_size_z_um

    offset = images.offsets.of_time_point(time_point)
    for position in experiment.positions.of_time_point(time_point):
        if abs(position.z - image_z_location) > max_px_from_slice:
            continue
        image_x = (position.x - offset.x) * x_scale
        image_y = (position.y - offset.y) * y_scale
        _draw_marker(image_2d_rgb, image_x, image_y)


#############
# XZ movies #
#############


def _generate_xz_movie(experiment: Experiment, y: int, channel: ImageChannel) -> ndarray:
    """Returns a TXZ array, slicing the images."""
    images = experiment.images
    positions = experiment.positions

    # Use time points with positions, except if there are none, then use time points with images
    time_points = list(positions.time_points())
    if len(time_points) == 0:
        time_points = list(images.time_points())

    image_size_zyx = images.image_loader().get_image_size_zyx()
    resolution = images.resolution()

    x_scale = _GLOBAL_SCALE
    z_scale = _GLOBAL_SCALE * (resolution.pixel_size_z_um / resolution.pixel_size_x_um)

    # Shape of array: TZXC
    output_array = numpy.zeros(
        (len(time_points), int(image_size_zyx[0] * z_scale), int(image_size_zyx[2] * x_scale), 3),
        dtype=numpy.uint8)
    temp_array = numpy.zeros(output_array.shape[1:3], dtype=numpy.uint8)  # Just a 2D, uncolored slice
    for i, time_point in enumerate(time_points):
        print(f"Working on time point {time_point.time_point_number()}...")

        # Draw image
        image_slice = _generate_xz_image(images, y, time_point, channel)
        if image_slice is not None:
            image_slice = bits.ensure_8bit(image_slice)
            cv2.resize(image_slice, dst=temp_array, dsize=(output_array[i].shape[1], output_array[i].shape[0]),
                       fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            output_array[i, :, :, 0] = temp_array
            output_array[i, :, :, 1] = temp_array
            output_array[i, :, :, 2] = temp_array

        # Draw positions
        _draw_xz_positions(output_array[i], experiment, time_point, x_scale, y, z_scale)

    return output_array


def _draw_xz_positions(image_2d_rgb: ndarray, experiment: Experiment, time_point: TimePoint, x_scale: float,
                       image_y_location: int, z_scale: float):
    """Draws the positions on the given 2d RGB image."""
    images = experiment.images
    resolution = images.resolution()
    max_px_from_slice = _MAX_DISTANCE_UM_FROM_SLICE / resolution.pixel_size_y_um

    offset = images.offsets.of_time_point(time_point)
    for position in experiment.positions.of_time_point(time_point):
        if abs(position.y - image_y_location) > max_px_from_slice:
            continue
        image_x = (position.x - offset.x) * x_scale
        image_z = (position.z - offset.z) * z_scale
        _draw_marker(image_2d_rgb, image_x, image_z)


def _generate_xz_image(images: Images, y: int, time_point: TimePoint, channel: ImageChannel) -> Optional[ndarray]:
    """Returns an xz image slice."""
    image_3d = images.get_image(time_point, channel)
    if image_3d is None:
        raise ValueError(f"No image found for {time_point} {channel}")
    image_y = int(y - image_3d.offset.y)
    if image_y < 0 or image_y >= image_3d.array.shape[1]:
        return None
    return image_3d.array[:, image_y, :]


#############
# YZ movies #
#############


def _generate_yz_movie(experiment: Experiment, x: int, channel: ImageChannel) -> ndarray:
    """Returns a TYZ array, slicing the images."""
    images = experiment.images
    positions = experiment.positions

    # Use time points with positions, except if there are none, then use time points with images
    time_points = list(positions.time_points())
    if len(time_points) == 0:
        time_points = list(images.time_points())

    image_size_zyx = images.image_loader().get_image_size_zyx()
    resolution = images.resolution()

    y_scale = _GLOBAL_SCALE
    z_scale = _GLOBAL_SCALE * (resolution.pixel_size_z_um / resolution.pixel_size_y_um)

    # Shape of array: TZYC
    output_array = numpy.zeros(
        (len(time_points), int(image_size_zyx[0] * z_scale), int(image_size_zyx[1] * y_scale), 3),
        dtype=numpy.uint8)
    temp_array = numpy.zeros(output_array.shape[1:3], dtype=numpy.uint8)  # Just a 2D, uncolored slice
    for i, time_point in enumerate(time_points):
        print(f"Working on time point {time_point.time_point_number()}...")

        # Draw image
        image_slice = _generate_yz_image(images, x, time_point, channel)
        if image_slice is not None:
            image_slice = bits.ensure_8bit(image_slice)
            cv2.resize(image_slice, dst=temp_array, dsize=(output_array[i].shape[1], output_array[i].shape[0]),
                       fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            output_array[i, :, :, 0] = temp_array
            output_array[i, :, :, 1] = temp_array
            output_array[i, :, :, 2] = temp_array

        # Draw positions
        _draw_yz_positions(output_array[i], experiment, time_point, x, y_scale, z_scale)

    return output_array


def _draw_yz_positions(image_2d_rgb: ndarray, experiment: Experiment, time_point: TimePoint, image_x_location: int,
                       y_scale: float, z_scale: float):
    """Draws the positions on the given 2d RGB image."""
    images = experiment.images
    resolution = images.resolution()
    max_px_from_slice = _MAX_DISTANCE_UM_FROM_SLICE / resolution.pixel_size_x_um

    offset = images.offsets.of_time_point(time_point)
    for position in experiment.positions.of_time_point(time_point):
        if abs(position.x - image_x_location) > max_px_from_slice:
            continue
        image_y = (position.y - offset.y) * y_scale
        image_z = (position.z - offset.z) * z_scale
        _draw_marker(image_2d_rgb, image_y, image_z)


def _generate_yz_image(images: Images, x: int, time_point: TimePoint, channel: ImageChannel) -> Optional[ndarray]:
    """Returns an yz image slice."""
    image_3d = images.get_image(time_point, channel)
    if image_3d is None:
        raise ValueError(f"No image found for {time_point} {channel}")
    image_x = int(x - image_3d.offset.x)
    if image_x < 0 or image_x >= image_3d.array.shape[2]:
        return None
    return image_3d.array[:, :, image_x]
