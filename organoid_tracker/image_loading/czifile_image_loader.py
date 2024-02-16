"""Image loader for CZI files."""
import os.path
from typing import Optional, Tuple, Union

import numpy
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.image_loading import _czi
from organoid_tracker.util.xml_wrapper import XmlWrapper, read_xml


def load_from_czi_file(experiment: Experiment, file: str, series_index: Union[str, int], min_time_point: int = 0,
                       max_time_point: int = 1000000000):
    """Sets up the experimental images for a LIF file that is not yet opened."""
    if not os.path.exists(file):
        print("Failed to load \"" + file + "\" - file does not exist")
        return

    load_from_czi_reader(experiment, file, _czi.CziFile(file), int(series_index), min_time_point, max_time_point)


def _read_resolution(metadata: XmlWrapper) -> Optional[ImageResolution]:
    resolution_zyx = _read_resolution_zyx_um(metadata)
    if resolution_zyx == (0, 0, 0):
        return None  # Couldn't read the resolution

    # I wonder if the time resolution is always stored like this in CZI files
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


def load_from_czi_reader(experiment: Experiment, file: str, reader: _czi.CziFile, serie_index: int,
                         min_time_point: int = 0,
                         max_time_point: int = 1000000000):
    """Sets up the experimental images for an already opened LIF file."""
    experiment.images.image_loader(_CziImageLoader(file, reader, serie_index, min_time_point, max_time_point))

    # Set up resolution
    metadata = read_xml(reader.metadata(raw=True))
    experiment.images.set_resolution(_read_resolution(metadata))

    # Generate an automatic name for the experiment
    file_name = os.path.basename(file)
    if file_name.lower().endswith(".czi"):
        file_name = file_name[:-4]
    experiment.name.provide_automatic_name(file_name + " #" + str(serie_index + 1))


class _CziImageLoader(ImageLoader):
    _file: str
    _reader: _czi.CziFile
    _series_index: int
    _location_to_subblock_mapping: ndarray  # Indexed as C, T, Z

    _min_time_point_number: int
    _max_time_point_number: int

    def __init__(self, file: str, reader: _czi.CziFile, series_index: int, min_time_point: int, max_time_point: int):
        self._file = file
        self._reader = reader
        self._series_index = series_index

        shape = reader.shape
        axes = reader.axes
        time_points = shape[axes.index("T")] if "T" in axes else 1

        if min_time_point is None:
            min_time_point = 0
        if max_time_point >= time_points:
            max_time_point = time_points - 1
        self._min_time_point_number = min_time_point
        self._max_time_point_number = max_time_point

        self._location_to_subblock_mapping = self._build_subblock_mapping(axes, series_index, shape)

    def _build_subblock_mapping(self, axes, series_index, shape) -> ndarray:
        """Builds a 3D array, containing the indices of self._reader.filtered_subblock_directory for each location."""
        z_size = shape[axes.index("Z")] if "Z" in axes else 1
        channel_count = shape[axes.index("C")] if "C" in axes else 1
        time_count = shape[axes.index("T")] if "T" in axes else 1
        location_to_subblock_mapping = numpy.zeros((channel_count, time_count, z_size), dtype=numpy.uint32)
        axes = self._reader.axes
        subblock_series_location = axes.index("S") if "S" in axes else -1
        subblock_time_location = axes.index("T") if "T" in axes else -1
        subblock_channel_location = axes.index("C") if "C" in axes else -1
        subblock_z_location = axes.index("Z") if "Z" in axes else -1
        for i, subblock in enumerate(self._reader.filtered_subblock_directory):
            subblock_series_index = subblock.start[subblock_series_location] if subblock_series_location > 0 else 0
            if subblock_series_index != series_index:
                continue
            subblock_time_index = subblock.start[subblock_time_location] if subblock_time_location > 0 else 0
            subblock_channel_index = subblock.start[subblock_channel_location] if subblock_channel_location > 0 else 0
            subblock_z_index = subblock.start[subblock_z_location] if subblock_z_location > 0 else 0
            location_to_subblock_mapping[subblock_channel_index, subblock_time_index, subblock_z_index] = i
        return location_to_subblock_mapping

    def first_time_point_number(self) -> int:
        """Gets the first time point for which images are available."""
        return self._min_time_point_number

    def last_time_point_number(self) -> int:
        """Gets the last time point (inclusive) for which images are available."""
        return self._max_time_point_number

    def get_channel_count(self) -> int:
        shape = self._reader.shape
        axes = self._reader.axes

        return shape[axes.index("C")] if "C" in axes else 1

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point_number \
                or time_point.time_point_number() > self._max_time_point_number:
            return None
        if image_channel.index_zero >= self.get_channel_count():
            return None

        image_shape_zyx = self.get_image_size_zyx()
        array = numpy.zeros(image_shape_zyx, dtype=self._reader.dtype)
        for z in range(array.shape[0]):
            subblock_index = self._location_to_subblock_mapping[
                image_channel.index_zero, time_point.time_point_number(), z]
            array[z] = self._reader.filtered_subblock_directory[subblock_index].data_segment()\
                        .data(resize=True, order=0).reshape(image_shape_zyx[1:])
        return array

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if time_point.time_point_number() < self._min_time_point_number \
                or time_point.time_point_number() > self._max_time_point_number:
            return None

        image_shape_zyx = self.get_image_size_zyx()
        if image_z < 0 or image_z >= image_shape_zyx[0]:
            return None

        if image_channel.index_zero < 0 or image_channel.index_zero >= self._location_to_subblock_mapping.shape[0]:
            return None

        subblock_index = self._location_to_subblock_mapping[
            image_channel.index_zero, time_point.time_point_number(), image_z]
        array = self._reader.filtered_subblock_directory[subblock_index].data_segment().data(resize=True, order=0)
        array = array.reshape(image_shape_zyx[1:])
        return array

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        shape = self._reader.shape
        axes = self._reader.axes

        x_size = shape[axes.index("X")] if "X" in axes else 1
        y_size = shape[axes.index("Y")] if "Y" in axes else 1
        z_size = shape[axes.index("Z")] if "Z" in axes else 1
        return z_size, y_size, x_size

    def copy(self) -> "_CziImageLoader":
        return _CziImageLoader(self._file, _czi.CziFile(self._file), self._series_index, self._min_time_point_number,
                               self._max_time_point_number)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file, str(self._series_index)

    def close(self):
        self._reader.close()


def read_czi_file(file_path: str) -> Tuple[_czi.CziFile, int]:
    """Reads a CZI file and returns the reader and the number of series."""
    reader = _czi.CziFile(file_path)
    axes = reader.axes
    series_count = reader.shape[axes.index("S")] if "S" in axes else 1
    return reader, series_count
