import os
from typing import Tuple, Optional, Dict, Any

import zarr
from numpy import ndarray
from zarr.core.attributes import Attributes
from zarr.storage import LocalStore, ZipStore

from organoid_tracker.core import TimePoint, UserError, image_coloring
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.images import ChannelDescription
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.util import unit_conversion

_SUPPORTED_AXIS = ("t", "c", "z", "y", "x")


def load_from_zarr_file(experiment: Experiment, file_name: str, min_time_point: int = -9999999, max_time_point: int = 9999999):
    """Loads image data from a ZARR file into the given experiment."""

    # Remove .zgroup or .zarray to get the container name
    if file_name.endswith(".zgroup") or file_name.endswith(".zarray"):
        file_name = file_name[:-7]  # Both strings are 7 characters long, so this works

    # Open Zarr container
    zarr_store = ZipStore(file_name, read_only=True) if file_name.lower().endswith(".zip") else LocalStore(
        file_name, read_only=True)
    zarr_group_or_array = zarr.open(zarr_store, mode="r")

    # Check what we just loaded - a ZARR Group or Array?
    if isinstance(zarr_group_or_array, zarr.Group):
        # We have an (OME-)Zarr group
        keys = list(zarr_group_or_array.keys())
        if len(keys) == 0:
            raise UserError("Empty ZARR file",
                            f"Found a ZARR group with no keys in file {file_name}, which we cannot handle.")

        # Check for a GEFF inside this ZARR (stored in "tracks")
        axes_names_geff = None
        if "tracks" in keys:
            # Found GEFF metadata!

            # Read the axis
            axes_names_geff = _read_axis_order_from_geff(zarr_group_or_array["tracks"])

            if not experiment.links.has_links():
                # We only load the tracking data if there isn't any yet - we don't want to just
                # overwrite any existing data

                from organoid_tracker.imaging import geff_io
                # Would be nice if we could just pass the already-open store
                geff_io.load_data_file(os.path.join(file_name, "tracks"), min_time_point=min_time_point,
                                       max_time_point=max_time_point, experiment=experiment)

            keys.remove("tracks")  # So that we don't try to read this as an image in the next step
        if len(keys) == 0:
            return

        # After reading the GEFF (if any), there are still other groups left
        # Hopefully these are the images

        # We currently only support one image in the OME-Zarr group, so we take the first one. If there are multiple, we ignore the others.
        key = keys[0]
        zarr_sub_entry = zarr_group_or_array[key]

        if isinstance(zarr_sub_entry, zarr.Array):
            # Found an array!
            axes_names = _read_axes_order_from_attrs(zarr_group_or_array.attrs)
            if axes_names is None:
                axes_names = axes_names_geff  # Try the ones from GEFF instead, if any
            if axes_names is None:
                axes_names = _guess_axes_order_from_shape(zarr_sub_entry)  # Make an educated guess

            experiment.images.image_loader(_ZarrImageLoader(file_name, axes_names, zarr_sub_entry, min_time_point, max_time_point))
            _set_experiment_name(experiment, file_name)
            _set_experiment_resolution(experiment, zarr_group_or_array.attrs)

            if key == "segmentation":
                # Set the appropriate colormap
                experiment.images.set_channel_description(ImageChannel(index_one=1),
                         ChannelDescription(channel_name="1", colormap=image_coloring.get_segmentation_colormap()))
        else:
            raise UserError("ZARR file too complex", "First entry in ZARR group was not an array - cannot handle this case.")
    elif isinstance(zarr_group_or_array, zarr.Array):
        # We just have a bare array, try to display it
        axes_names = _guess_axes_order_from_shape(zarr_group_or_array)
        experiment.images.image_loader(_ZarrImageLoader(file_name, axes_names, zarr_group_or_array, min_time_point, max_time_point))
        _set_experiment_name(experiment, file_name)
    else:
        # Don't know what happened here
        raise UserError("Unsupported ZARR", f"Found unsupported entry: {zarr_group_or_array}")


def _guess_axes_order_from_shape(zarr_array: zarr.Array) -> str:
    """This method just guesses what the shape of the array could be. Only used in cases where we can't read the
     metadata."""
    shape = zarr_array.shape
    if len(shape) == 2:
        return "yx"
    if len(shape) == 3:
        return "tyx"
    if len(shape) == 4:
        return "tzyx"
    if len(shape) == 5:
        return "tczyx"
    raise UserError("Unsupported ZARR", f"Found a ZARR array with shape {shape}, which we cannot handle.")


def _read_axis_order_from_geff(tracks_group: zarr.Group) -> str:
    """Expects a ZARR Group that is in the format of a GEFF. Then extracts the axis order."""
    axes = tracks_group.metadata.attributes["geff"]["axes"]
    axes_names = list()
    for axis in axes:
        if axis["type"] == "time":
            axes_names.append("t")
        elif axis["type"] == "space":
            if axis["name"] not in _SUPPORTED_AXIS:
                raise UserError("Unsupported axis", f"Found unsupported axis in GEFF metadata: \"{axis['name']}\".")
            axes_names.append(axis["name"])
        else:
            raise UserError("Unsupported axis type", f"Found unsupported axis type in GEFF metadata: \"{axis['type']}\".")
    return "".join(axes_names)


def _read_axes_order_from_attrs(attributes: Attributes) -> Optional[str]:
    """Reads the attributes of a ZARR file, to figure out the axis order.
    Returned string uses letters from _SUPPORTED_AXIS."""
    if "multiscales" not in attributes:
        return None  # Right now, we only support the axes metadata in the "multiscales" format

    multiscales_meta = attributes["multiscales"][0]
    if "axes" not in multiscales_meta:
        return None

    multiscales_axes_metadata = multiscales_meta["axes"]

    axes_names = []
    for axis in multiscales_axes_metadata:
        axis_name = axis["name"].lower()
        if axis_name not in _SUPPORTED_AXIS:
            raise UserError("Unsupported axis", f"Found unsupported axis: \"{axis_name}\".")
        axes_names.append(axis_name)
    return "".join(axes_names)


def _set_experiment_name(experiment: Experiment, file_name: str):
    """Sets an automatic name of the experiment based on the file name of the ZARR folder."""
    file_name = os.path.basename(file_name)
    if file_name.lower().endswith(".zarr"):
        file_name = file_name[:-len(".zarr")]
    experiment.name.provide_automatic_name(file_name)


def _set_experiment_resolution(experiment: Experiment, attributes: Attributes):
    if "multiscales" not in attributes:
        return  # Right now, we only support the axes metadata in the "multiscales" format

    multiscales_meta = attributes["multiscales"][0]
    if "axes" not in multiscales_meta or "datasets" not in multiscales_meta:
        return

    multiscales_axes_metadata = multiscales_meta["axes"]
    dataset_metadata = multiscales_meta["datasets"][0]
    if "coordinateTransformations" not in dataset_metadata:
        return
    scales: list[float] = []
    for coordinate_transform in dataset_metadata["coordinateTransformations"]:
        if "scale" in coordinate_transform and len(coordinate_transform["scale"]) > 0:
            scales = coordinate_transform["scale"]
    if len(scales) == 0:
        return  # Didn't find a valid scale

    x_resolution_um = None  # X and Y resolution are required, for Z and T we can use a default value
    y_resolution_um = None
    z_resolution_um = 0
    time_resolution_minutes = 0
    for axis, scale in zip(multiscales_axes_metadata, scales):
        axis_name = axis["name"].lower()
        if "unit" not in axis:
            continue  # No unit specified, don't know what to do with the number
        axis_unit = axis["unit"].lower()
        if axis_name in "xyz":
            if axis_unit not in unit_conversion.MULTIPLICATION_FACTOR_TO_MICROMETERS:
                continue  # Can't handle this unit, so we skip it
            if axis_name == "x":
                x_resolution_um = scale * unit_conversion.MULTIPLICATION_FACTOR_TO_MICROMETERS[axis_unit]
            elif axis_name == "y":
                y_resolution_um = scale * unit_conversion.MULTIPLICATION_FACTOR_TO_MICROMETERS[axis_unit]
            elif axis_name == "z":
                z_resolution_um = scale * unit_conversion.MULTIPLICATION_FACTOR_TO_MICROMETERS[axis_unit]
        elif axis_name == "t":
            if axis_unit not in unit_conversion.MULTIPLICATION_FACTOR_TO_MINUTES:
                continue  # Can't handle this unit, so we skip it
            time_resolution_minutes = scale * unit_conversion.MULTIPLICATION_FACTOR_TO_MINUTES[axis_unit] * 60  # Convert to minutes
    if x_resolution_um is not None and y_resolution_um is not None:
        # Got enough information, we can set a resolution
        resolution = ImageResolution(x_resolution_um, y_resolution_um, z_resolution_um, time_resolution_minutes)
        experiment.images.set_resolution(resolution)


class _ZarrImageLoader(ImageLoader):

    _file_name: str

    _raw_zarr_array: zarr.Array

    _axes_names: str
    _axes_sizes: Tuple[int, ...]

    _min_available_time_point_number: int
    _max_available_time_point_number: int

    _z_count: int
    _channel_count: int

    def __init__(self, file_name: str, axes_names: str, array: zarr.Array, min_time_point: Optional[int] = None, max_time_point: Optional[int] = None):
        self._file_name = file_name
        self._axes_names = axes_names
        self._raw_zarr_array = array

        self._axes_sizes = self._raw_zarr_array.shape

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

        return self._raw_zarr_array[tuple(indices)]

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

        return self._raw_zarr_array[tuple(indices)]

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
        return self._file_name, "0"

    def copy(self) -> "ImageLoader":
        return _ZarrImageLoader(self._file_name, self._min_available_time_point_number, self._max_available_time_point_number)

    def close(self):
        self._raw_zarr_array.store.close()
