from typing import Optional, Tuple, List, Any

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from numpy import ndarray


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

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        if time_point.time_point_number() == 1 and image_channel.index_one == 1:
            return self._image
        return None

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        if time_point.time_point_number() == 1 and image_channel.index_one == 1:
            if image_z >= 0 and image_z < self._image.shape[0]:
                return self._image[image_z]
        return None

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._image.shape

    def first_time_point_number(self) -> Optional[int]:
        return 1

    def last_time_point_number(self) -> Optional[int]:
        return 1

    def get_channel_count(self) -> int:
        return 1

    def copy(self) -> "ImageLoader":
        return SingleImageLoader(self._image, self._file_name)

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._file_name, ""
