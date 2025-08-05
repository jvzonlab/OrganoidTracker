"""Image loader for CZI files."""
import os.path
from threading import Lock
from typing import Optional, Tuple, Union, Dict

import numpy
from aicspylibczi import CziFile
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.util.xml_wrapper import XmlWrapper, read_xml


def load_from_czi_file(experiment: Experiment, file: str, series_index_one: Union[str, int], min_time_point: int = 0,
                       max_time_point: int = 1000000000):
    """Sets up the experimental images for a LIF file that is not yet opened."""
    if not os.path.exists(file):
        print("Failed to load \"" + file + "\" - file does not exist")
        return

    load_from_czi_reader(experiment, file, CziFile(file), int(series_index_one), min_time_point, max_time_point)


def _read_resolution(metadata: XmlWrapper) -> Optional[ImageResolution]:
    resolution_zyx = _read_resolution_zyx_um(metadata)
    if resolution_zyx == (0, 0, 0):
        return None  # Couldn't read the resolution

    # I wonder how we're supposed to read the time resolution. This is one way:
    time_resolution_value_s = metadata["Metadata"]["Information"]["Image"]["Dimensions"]["T"]["Positions"]["Interval"]["Increment"].value_float()

    if time_resolution_value_s is None:
        # This appears to be another way
        time_resolution = metadata["Metadata"]["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"]["TimeSeriesSetup"]\
            ["Switches"]["Switch"]["SwitchAction"]["SetIntervalAction"]["Interval"]["TimeSpan"]
        time_resolution_value_s = 0
        if time_resolution["DefaultUnitFormat"].value_str() == "ms":
            time_resolution_value_ms = time_resolution["Value"].value_float()
            if time_resolution_value_ms is not None:
                time_resolution_value_s = time_resolution_value_ms / 1000

    return ImageResolution(resolution_zyx[2], resolution_zyx[1], resolution_zyx[0], time_resolution_value_s / 60)


def _read_resolution_zyx_um(metadata: XmlWrapper) -> Tuple[float, float, float]:
    resolution = metadata["Metadata"]["Scaling"]["Items"]

    x_res_um = 0
    y_res_um = 0
    z_res_um = 0
    for element in resolution:
        if element.attr_str("Id") == "X":
            x_res_um = element["Value"].value_float() * 1_000_000
        elif element.attr_str("Id") == "Y":
            y_res_um = element["Value"].value_float() * 1_000_000
        elif element.attr_str("Id") == "Z":
            z_res_um = element["Value"].value_float() * 1_000_000
    return z_res_um, y_res_um, x_res_um


def load_from_czi_reader(experiment: Experiment, file: str, reader: CziFile, serie_index: int,
                         min_time_point: int = 0,
                         max_time_point: int = 1000000000):
    """Sets up the experimental images for an already opened LIF file."""
    experiment.images.image_loader(_CziImageLoader(file, reader, serie_index, min_time_point, max_time_point))

    # Set up resolution
    metadata = read_xml(reader.reader.read_meta())
    experiment.images.set_resolution(_read_resolution(metadata))

    # Generate an automatic name for the experiment
    file_name = os.path.basename(file)
    if file_name.lower().endswith(".czi"):
        file_name = file_name[:-4]
    experiment.name.provide_automatic_name(file_name + " #" + str(serie_index))


class _CziImageLoader(ImageLoader):
    _file: str
    _czi_file: CziFile
    _czi_lock: Lock  # Acquired from the public (outer) methods
    _series_index: int  # Series index, starts at 1

    _min_time_point_number: int
    _max_time_point_number: int

    def __init__(self, file: str, reader: CziFile, series_index: int, min_time_point: int, max_time_point: int):
        self._file = file
        self._czi_file = reader
        self._czi_lock = Lock()
        self._series_index = series_index

        dims_shape = self._get_dims_shape()
        min_time_point_file, max_time_point_file_exclusive = dims_shape.get("T", (0, 1))

        if min_time_point is None or min_time_point < min_time_point_file:
            min_time_point =min_time_point_file
        if max_time_point >= max_time_point_file_exclusive:
            max_time_point = max_time_point_file_exclusive - 1
        self._min_time_point_number = min_time_point
        self._max_time_point_number = max_time_point

    def _get_dims_shape(self) -> Dict[str, Tuple[int, int]]:
        """Returns the dimensions and their shapes."""
        for dims_shape in self._czi_file.get_dims_shape():
            min_s, max_s_exclusive = dims_shape["S"]
            if min_s <= self._series_index < max_s_exclusive:
                return dims_shape
        raise ValueError(f"Series index {self._series_index} not found in CZI file {self._file}.")

    def first_time_point_number(self) -> int:
        """Gets the first time point for which images are available."""
        return self._min_time_point_number

    def last_time_point_number(self) -> int:
        """Gets the last time point (inclusive) for which images are available."""
        return self._max_time_point_number

    def get_channel_count(self) -> int:
        dim_shape = self._get_dims_shape()
        if "C" not in dim_shape:
            return 1
        min_c, max_c_exclusive = dim_shape["C"]

        return max_c_exclusive - min_c  # So if min_c is 0 and max_c_exclusive is 3, we have 3 channels: (0, 1, 2)

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point_number \
                or time_point.time_point_number() > self._max_time_point_number:
            return None

        channel_count = self.get_channel_count()
        if image_channel.index_zero < 0 or image_channel.index_zero >= channel_count:
            return None

        # Build an image query in such a way that we avoid the "The coordinates are overspecified" error is avoided,
        # by only querying the dimensions that are available in the CZI file.
        available_keys = self._get_dims_shape().keys()
        image_query = {"S": self._series_index,
                       "T": time_point.time_point_number(),
                       "C": image_channel.index_zero}
        for dim in list(image_query.keys()):
            if dim not in available_keys:
                del image_query[dim]

        with self._czi_lock:
            array, dims_and_size = self._czi_file.read_image(**image_query)

        # Check returned dimensions and then squeeze the array
        for dim, size in dims_and_size:
            if size > 1 and dim not in ("X", "Y", "Z"):
                raise ValueError(f"Got a size larger than 1 for a dimension that is not X, Y or Z: [{dim}={size}]")
        squeezed_array = numpy.squeeze(array)

        # Make sure we don't accidentally return a 2D array
        if squeezed_array.ndim == 2:
            return squeezed_array[numpy.newaxis, :, :]
        return squeezed_array

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point_number \
                or time_point.time_point_number() > self._max_time_point_number:
            return None

        image_shape_zyx = self.get_image_size_zyx()
        if image_z < 0 or image_z >= image_shape_zyx[0]:
            return None

        channel_count = self.get_channel_count()
        if image_channel.index_zero < 0 or image_channel.index_zero >= channel_count:
            return None

        # Build an image query in such a way that we avoid the "The coordinates are overspecified" error is avoided,
        # by only querying the dimensions that are available in the CZI file.
        available_keys = self._get_dims_shape().keys()
        image_query = {"S": self._series_index,
                       "T": time_point.time_point_number(),
                       "C": image_channel.index_zero,
                       "Z": image_z}
        for dim in list(image_query.keys()):
            if dim not in available_keys:
                del image_query[dim]

        with self._czi_lock:
            array, dims_and_size = self._czi_file.read_image(**image_query)

        # Check returned dimensions and then squeeze the array
        for dim, size in dims_and_size:
            if size > 1 and dim not in ("X", "Y"):
                raise ValueError(f"Got a size larger than 1 for a dimension that is not X or Y: [{dim}={size}]")
        return numpy.squeeze(array)

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        size = self._czi_file.size
        dims = self._czi_file.dims

        x_size = size[dims.index("X")] if "X" in dims else 1
        y_size = size[dims.index("Y")] if "Y" in dims else 1
        z_size = size[dims.index("Z")] if "Z" in dims else 1
        return z_size, y_size, x_size

    def copy(self) -> "_CziImageLoader":
        return _CziImageLoader(self._file, CziFile(self._file), self._series_index, self._min_time_point_number,
                               self._max_time_point_number)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file, str(self._series_index)

    def close(self):
        with self._czi_lock:
            # Suppress the warning - we cannot close the file otherwise
            # noinspection PyProtectedMember
            self._czi_file._bytes.close()


def read_czi_file(file_path: str) -> Tuple[CziFile, int, int]:
    """Reads a CZI file and returns the reader and the available series range (inclusive)."""
    reader = CziFile(file_path)
    available_series = set()
    for dim_shape in  reader.get_dims_shape():
        s_min, s_max_exclusive = dim_shape["S"]
        for s in range(s_min, s_max_exclusive):
            available_series.add(s)

    available_series = list(available_series)
    available_series.sort()
    return reader, min(available_series), max(available_series)
