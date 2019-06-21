from typing import Optional, Tuple, List
from xml.dom.minidom import Element

from numpy import ndarray

from autotrack.core import TimePoint
from autotrack.core.image_loader import ImageLoader, ImageChannel
from autotrack.core.images import Images
from autotrack.core.resolution import ImageResolution
from autotrack.imaging import lif


def load_from_lif_file(images: Images, file: str, series_name: str, min_time_point: int = 0,
                       max_time_point: int = 1000000000):
    """Sets up the experimental images for a LIF file that is not yet opened."""
    reader = lif.Reader(file)

    # Find index of series
    series_index = None
    for index, header in enumerate(reader.getSeriesHeaders()):
        if header.getName() == series_name:
            series_index = index
    if series_index is None:
        raise ValueError("No series matched the given name. Available names: "
                         + str([header.getName() for header in reader.getSeriesHeaders()]))

    load_from_lif_reader(images, file, reader, series_index, min_time_point, max_time_point)


def load_from_lif_reader(images: Images, file: str, reader: lif.Reader, serie_index: int, min_time_point: int = 0,
                         max_time_point: int = 1000000000):
    """Sets up the experimental images for an already opened LIF file."""
    images.image_loader(_LifImageLoader(file, reader, serie_index, min_time_point, max_time_point))
    serie_header = reader.getSeriesHeaders()[serie_index]
    dimensions = serie_header.getDimensions()
    images.set_resolution(_dimensions_to_resolution(dimensions))


def _dimensions_to_resolution(dimensions: List[Element]) -> ImageResolution:
    pixel_size_x_um = 0
    pixel_size_y_um = 0
    pixel_size_z_um = 0
    time_point_interval_m = 0

    for dimension in dimensions:
        axis = int(dimension.getAttribute("DimID"))
        axis_name = lif.dimName[axis]
        total_length = float(dimension.getAttribute("Length"))
        number_of_elements = int(dimension.getAttribute("NumberOfElements"))
        unit_length = 0 if number_of_elements == 1 else total_length / (number_of_elements - 1)
        #             ^ no resolution exists if you only have one element.
        # Example: what is the time resolution if you have just 1 time point?

        if axis_name == "X" or axis_name == "Y" or axis_name == "Z":
            if dimension.getAttribute("Unit") != "m":
                raise ValueError("Unknown unit: " + dimension.getAttribute("Unit"))
            unit_length_um = unit_length * 1000000  # From m to um
            if axis_name == "X":
                pixel_size_x_um = unit_length_um
            elif axis_name == "Y":
                pixel_size_y_um = unit_length_um
            elif axis_name == "Z":
                pixel_size_z_um = unit_length_um
        elif axis_name == "T":
            if dimension.getAttribute("Unit") != "s":
                raise ValueError("Unknown unit: " + dimension.getAttribute("Unit"))
            time_point_interval_m = unit_length / 60
        else:
            raise ValueError("Unknown DimID: " + axis_name)

    return ImageResolution(pixel_size_x_um, pixel_size_y_um, pixel_size_z_um, time_point_interval_m)


class _IndexedChannel(ImageChannel):

    index: int

    def __init__(self, index: int):
        self.index = index


class _LifImageLoader(ImageLoader):

    _file: str
    _serie: lif.Serie
    _serie_index: int

    _min_time_point_number: int
    _max_time_point_number: int
    _inverted_z: bool = False

    _channels: List[_IndexedChannel]

    def __init__(self, file: str, reader: lif.Reader, serie_index: int, min_time_point: int, max_time_point: int):
        self._file = file
        self._serie = reader.getSeries()[serie_index]
        self._channels = [_IndexedChannel(i) for i, channel in enumerate(self._serie.getChannels())]
        self._serie_index = serie_index

        for dimension in self._serie.getDimensions():
            axis = int(dimension.getAttribute("DimID"))
            axis_name = lif.dimName[axis]
            if axis_name == "Z":
                if dimension.getAttribute("Length")[0] == "-":
                    self._inverted_z = True

        if min_time_point is None:
            min_time_point = 0
        if max_time_point > self._serie.getNbFrames() - 1:
            max_time_point = self._serie.getNbFrames() - 1
        self._min_time_point_number = min_time_point
        self._max_time_point_number = max_time_point

    def first_time_point_number(self) -> int:
        """Gets the first time point for which images are available."""
        return self._min_time_point_number

    def last_time_point_number(self) -> int:
        """Gets the last time point (inclusive) for which images are available."""
        return self._max_time_point_number

    def get_channels(self) -> List[ImageChannel]:
        return self._channels

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        """Loads an image, usually from disk. Returns None if there is no image for this time point."""
        if time_point.time_point_number() < self._min_time_point_number\
                or time_point.time_point_number() > self._max_time_point_number:
            return None
        if not isinstance(image_channel, _IndexedChannel):
            return None

        array = self._serie.getFrame(time_point.time_point_number())
        if self._inverted_z:
            return array[::-1, image_channel.index]
        else:
            return array[:, image_channel.index]

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        dimensions: List[Element] = self._serie.getDimensions()
        x_size = dimensions[0].getAttribute("NumberOfElements")
        y_size = dimensions[1].getAttribute("NumberOfElements")
        z_size = dimensions[2].getAttribute("NumberOfElements")
        return int(z_size), int(y_size), int(x_size)

    def copy(self) -> "_LifImageLoader":
        return _LifImageLoader(self._file, lif.Reader(self._file), self._serie_index, self._min_time_point_number,
                               self._max_time_point_number)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file, self._serie.getName()
