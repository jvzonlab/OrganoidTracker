from typing import Tuple, Optional

import numpy
from numpy import ndarray
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Node

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel


_SUPPORTED_AXIS = ("t", "c", "z", "y", "x")


def load_from_zarr_file(experiment: Experiment, container: str, min_time_point: Optional[int] = None, max_time_point: Optional[int] = None):
    """Loads image data from a Zarr file into the given experiment."""
    experiment.images.image_loader(_ZarrImageLoader(container, min_time_point, max_time_point))


class _ZarrImageLoader(ImageLoader):

    _file_name: str
    _zarr_store: Reader

    _node_index: int = 0
    _image_node: Node

    _axes_names: str
    _axes_sizes: Tuple[int, ...]

    _min_available_time_point_number: int
    _max_available_time_point_number: int

    _z_count: int
    _channel_count: int

    def __init__(self, file_name: str, min_time_point: Optional[int] = None, max_time_point: Optional[int] = None):
        # Remove .zgroup to get the container name
        if file_name.endswith(".zgroup"):
            file_name = file_name[:-7]
        self._file_name = file_name

        # Open Zarr container
        zarr_location = parse_url(file_name)
        if zarr_location is None:
            raise UserError("Not a Zarr container", f"The file '{file_name}' is not a valid Zarr container.")
        self._zarr_store = Reader(parse_url(file_name))

        # Find image nodes
        nodes = list(self._zarr_store())
        if len(nodes) == 0:
            self.close()
            raise UserError("No image data", f"No image data found in Zarr file '{file_name}'")
        self._image_node = nodes[self._node_index]

        # Find axes names and sizes
        self._axes_names = ""
        self._axes_sizes = self._image_node.data[0].shape
        for ax in self._image_node.metadata["axes"]:
            ax_name = ax["name"]
            if ax_name not in _SUPPORTED_AXIS:
                self.close()
                raise UserError("Unsupported axis", f"We cannot read Zarr files with an axis '{ax_name}'."
                                                    f"Only the axes '{''.join(_SUPPORTED_AXIS)}' are supported.")
            self._axes_names += ax_name

        # Find available time points
        self._min_available_time_point_number = 0
        if "t" in self._axes_names:
            self._max_available_time_point_number = self._axes_sizes[self._axes_names.index("t")] - 1
        else:
            self._max_available_time_point_number = 0  # No time axis, so first and last time point are the same
        if min_time_point is not None and min_time_point > self._min_available_time_point_number:
            self._min_available_time_point_number = min_time_point
        if max_time_point is not None and max_time_point < self._max_available_time_point_number:
            self._max_available_time_point_number = max_time_point

        # Find sizes in z and channel
        self._z_count = 1 if "z" not in self._axes_names else self._axes_sizes[self._axes_names.index("z")]
        self._channel_count = 1 if "c" not in self._axes_names else self._axes_sizes[self._axes_names.index("c")]

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        # Check bounds
        if time_point.time_point_number() < self._min_available_time_point_number or time_point.time_point_number() > self._max_available_time_point_number:
            return None
        if image_channel.index_zero < 0 or image_channel.index_zero >= self._channel_count:
            return None

        # Build indices array
        indices = []
        for axis_name in self._axes_names:
            if axis_name == "t":
                indices.append(time_point.time_point_number())
            elif axis_name == "c":
                indices.append(image_channel.index_zero)
            elif axis_name in ("z", "y", "x"):
                indices.append(slice(None))

        return numpy.asarray(self._image_node.data[0][tuple(indices)])

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        # Check bounds
        if time_point.time_point_number() < self._min_available_time_point_number or time_point.time_point_number() > self._max_available_time_point_number:
            return None
        if image_z < 0 or image_z >= self._z_count:
            return None
        if image_channel.index_zero < 0 or image_channel.index_zero >= self._channel_count:
            return None

        # Build indices array
        indices = []
        for axis_name in self._axes_names:
            if axis_name == "t":
                indices.append(time_point.time_point_number())
            elif axis_name == "c":
                indices.append(image_channel.index_zero)
            elif axis_name == "z":
                indices.append(image_z)
            elif axis_name in ("y", "x"):
                indices.append(slice(None))

        return numpy.asarray(self._image_node.data[0][tuple(indices)])

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        y_size = self._axes_sizes[self._axes_names.index("y")]
        x_size = self._axes_sizes[self._axes_names.index("x")]
        return self._z_count, y_size, x_size

    def first_time_point_number(self) -> Optional[int]:
        return self._min_available_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        return self._max_available_time_point_number

    def get_channel_count(self) -> int:
        return self._channel_count

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file_name, str(self._node_index)

    def copy(self) -> "ImageLoader":
        return _ZarrImageLoader(self._file_name, self._min_available_time_point_number, self._max_available_time_point_number)

    def close(self):
        self._zarr_store.zarr.store.close()
        self._zarr_store = None
