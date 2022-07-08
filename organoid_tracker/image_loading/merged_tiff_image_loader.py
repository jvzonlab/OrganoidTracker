import os.path
from threading import Lock
from typing import Tuple, Any, List, Optional

import numpy
import tifffile
from numpy.core.multiarray import ndarray
from tifffile import TiffFile, TiffPageSeries

from organoid_tracker.core import TimePoint, max_none, min_none, UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.resolution import ImageResolution


def load_from_tif_file(experiment: Experiment, file: str, min_time_point: Optional[int] = None,
                       max_time_point: Optional[int] = None):
    """Creates an image loader for the individual images in the TIF file."""
    image_loader = _MergedTiffImageLoader(file, min_time_point, max_time_point)
    experiment.images.image_loader(image_loader)

    # Update resolution
    try:
        experiment.images.resolution()  # Tests if a resolution is already stored
    except UserError:
        # No resolution stored. Guess the resolution from the images
        resolution = image_loader.guess_resolution()
        if resolution is not None:
            experiment.images.set_resolution(resolution)

    # Update experiment name
    experiment.name.provide_automatic_name(image_loader.get_suggested_experiment_name())


class _MergedTiffImageLoader(ImageLoader):
    """Reader for huge TIF files, where an entire time series has been stored in a single file."""

    @staticmethod
    def _init_channels(axes: str, shape: Tuple[int, ...]) -> List[ImageChannel]:
        """Returns a list of all available channels."""
        channel_count = 1
        try:
            channel_index = axes.index("C")
            channel_count = shape[channel_index]
        except ValueError:
            pass  # No channel axis

        return [ImageChannel(index_zero=i) for i in range(channel_count)]

    @staticmethod
    def _init_image_size_zyx(axes: str, shape: Tuple[int, ...]) -> Tuple[int, int, int]:
        x_size = shape[axes.index("X")]
        y_size = shape[axes.index("Y")]
        try:
            z_size = shape[axes.index("Z")]
        except ValueError:
            z_size = 1  # No Z-axis
        return z_size, y_size, x_size

    @staticmethod
    def _get_highest_time_point(axes: str, shape: Tuple[int, ...]) -> int:
        """Gets the highest time point in the time series."""
        try:
            time_index = axes.index("T")
            return shape[time_index] - 1
        except ValueError:
            return 0  # No time axis

    # These are not thread safe, so we need to protect them behind a lock
    _tiff: TiffFile
    _tiff_series: TiffPageSeries
    _tiff_lock: Lock  # Acquired from the public (outer) methods

    _file_name: str
    _axes: str
    _shape: Tuple[int, ...]
    _is_rgb: bool = False
    _channels: List[ImageChannel]
    _image_size_zyx: Tuple[int, int, int]
    _min_time_point_number: int
    _max_time_point_number: int

    def __init__(self, file_name: str, min_time_point_number: Optional[int], max_time_point_number: Optional[int]):
        self._file_name = file_name

        self._tiff = TiffFile(file_name)
        self._tiff_series = self._tiff.series[0]
        self._tiff_lock = Lock()

        self._axes = self._tiff_series.axes
        self._shape = self._tiff_series.shape

        if self._axes[-2:] != "YX":
            self._tiff.close()
            raise UserError("Unsupported image", "We can only read TIF-stacks of black-and-white images, not RGB(A)"
                                                 " stacks.\nThis data format (" + self._axes + ") is therefore not supported.")

        self._channels = self._init_channels(self._axes, self._shape)
        self._image_size_zyx = self._init_image_size_zyx(self._axes, self._shape)
        self._min_time_point_number = max_none(0, min_time_point_number)
        self._max_time_point_number = min_none(self._get_highest_time_point(self._axes, self._shape),
                                               max_time_point_number)

    def guess_resolution(self) -> Optional[ImageResolution]:
        with self._tiff_lock:
            tags: TiffTags = self._tiff_series.pages[0].tags
        if "XResolution" not in tags or "YResolution" not in tags:
            return None
        x_res = tags["XResolution"].value
        y_res = tags["YResolution"].value
        x_um_px = x_res[1] / x_res[0]
        y_um_px = y_res[1] / x_res[0]
        z_um_px = 0
        t_minutes_tp = 0
        imagej_data = self._tiff.imagej_metadata
        if imagej_data is not None:
            if "spacing" in imagej_data and "unit" in imagej_data and imagej_data["unit"] == "micron":
                z_um_px = imagej_data["spacing"]
            if "finterval" in imagej_data:
                t_minutes_tp = imagej_data["finterval"] / 60
        return ImageResolution(x_um_px, y_um_px, z_um_px, t_minutes_tp)

    def _has_wrong_page_count(self) -> bool:
        """Some files (over 2 GB) have an apparently incorrect page count. This method returns True if that is the case.
        In that case, more low-level page reading functions need to be used."""
        expected_page_count = numpy.product(self._shape[0:-2])  # Every page is a 2D image
        page_count = len(self._tiff_series.pages)
        return page_count == 1 and expected_page_count > 1

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() < self.first_time_point_number() \
                or time_point.time_point_number() > self.last_time_point_number():
            return None
        if image_channel.index_zero >= len(self._channels):
            return None  # Invalid channel

        with self._tiff_lock:
            out = tifffile.create_output(None, self._image_size_zyx, self._tiff_series.dtype)
            for z in range(self._image_size_zyx[0]):
                self._get_2d_image_array(time_point.time_point_number(), image_channel.index, z, out[z])
        return out

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if time_point.time_point_number() < self.first_time_point_number() \
                or time_point.time_point_number() > self.last_time_point_number():
            return None  # Time out of range
        if image_channel.index_zero >= len(self._channels):
            return None  # Invalid channel
        if image_z < 0 or image_z >= self._image_size_zyx[0]:
            return None  # Z out of range

        with self._tiff_lock:
            out = tifffile.create_output(None, self._image_size_zyx[1:], self._tiff_series.dtype)
            self._get_2d_image_array(time_point.time_point_number(), image_channel.index, image_z, out)
        return out

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._image_size_zyx

    def first_time_point_number(self) -> Optional[int]:
        return self._min_time_point_number

    def last_time_point_number(self) -> Optional[int]:
        return self._max_time_point_number

    def get_channel_count(self) -> int:
        return len(self._channels)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file_name, ""

    def copy(self) -> "ImageLoader":
        return _MergedTiffImageLoader(self._file_name, self._min_time_point_number, self._max_time_point_number)

    def _get_offset(self, t: int, c: int, z: int) -> int:
        """Gets the pixel offset for the given 2D image."""
        offset = self._get_2d_page_number(t, c, z) * self._image_size_zyx[1] * self._image_size_zyx[2]
        return int(offset + self._tiff_series.offset)

    def _get_2d_page_number(self, t: int, c: int, z: int) -> int:
        """Gets the page number for the given 2D image.

        Note: this method must be called from within a synchronized block."""
        skip_axes = 3 if self._is_rgb else 2
        page = 0
        for i in range(len(self._axes) - skip_axes):
            axis = self._axes[i]
            multiplier = numpy.product(self._shape[i + 1:-skip_axes], dtype=numpy.uint64)
            if axis == "T":
                page += t * multiplier
            if axis == "Z":
                page += z * multiplier
            if axis == "C":
                page += c * multiplier
        return int(page)

    def _get_2d_image_array(self, t: int, c: int, z: int, out: ndarray):
        """Reads a 2D image array into the given output array. No range checks are performed. Make sure that the
        out array is created using tifffile.create_output(...).

        Note: this method must be called from within a synchronized block."""
        if not self._has_wrong_page_count():
            # Page count is correct - use high-level API
            page = self._get_2d_page_number(t, c, z)
            self._tiff.asarray(key=page, out=out)
        else:
            # Need to fiddle with bytes :(. Irfanview also has trouble with these files, tifffile is not the only one.
            offset = self._get_offset(t, c, z)
            shape_2d = self._shape[-2:]
            type_code = self._tiff.byteorder + self._tiff_series.dtype.char
            self._tiff.filehandle.seek(offset)
            self._tiff.filehandle.read_array(type_code, numpy.product(shape_2d), out=out)

    def get_suggested_experiment_name(self) -> str:
        """Gets the suggested experiment name. Returns "nd799xy08" for "C:/Images/nd799xy08.tif"."""
        file_name = os.path.basename(self._file_name)
        if file_name.lower().endswith(".tif"):
            return file_name[0:-4]
        if file_name.lower().endswith(".tiff"):
            return file_name[0:-5]
        return file_name
