from typing import Optional, Tuple, List

from autotrack.core import TimePoint
from autotrack.core.image_loader import ImageLoader, ImageChannel
from numpy import ndarray

_ONLY_CHANNEL = ImageChannel()


class SingleImageLoader(ImageLoader):
    """An image loader that just displays a single 3D array. Useful if you're writing a quick script."""

    _image: ndarray
    _file_name: str

    def __init__(self, array: ndarray, file_name: str = ""):
        """The file name is only used for serialize_to_config()"""
        if len(array.shape) != 3:
            raise ValueError("Can only handle 3D images")
        self._image = array
        self._file_name = file_name

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() == 1 and image_channel is _ONLY_CHANNEL:
            return self._image
        return None

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._image.shape

    def first_time_point_number(self) -> Optional[int]:
        return 1

    def last_time_point_number(self) -> Optional[int]:
        return 1

    def get_channels(self) -> List[ImageChannel]:
        return [_ONLY_CHANNEL]

    def copy(self) -> "ImageLoader":
        return SingleImageLoader(self._image, self._file_name)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file_name, ""
