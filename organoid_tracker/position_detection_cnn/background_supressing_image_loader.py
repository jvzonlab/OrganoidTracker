from typing import Optional, Tuple, List

import mahotas
from numpy import ndarray

from organoid_tracker.core import TimePoint
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel


class BackgroundSupressingImageLoader(ImageLoader):
    """Supresses background noise, which is useful to make the images of the Organoid Confocal look more like the images
    of the Spar Multiphoton Confocal. This is useful for using the network trained on the Multiphoton on the Organoid
    Confocal."""
    _internal: ImageLoader

    def __init__(self, internal: ImageLoader):
        self._internal = internal

    def copy(self) -> "ImageLoader":
        return BackgroundSupressingImageLoader(self._internal.copy())

    def get_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        array = self._internal.get_image_array(time_point, image_channel)
        T_otsu = mahotas.otsu(array)
        thresholded_1 = array > T_otsu
        thresholded_2 = mahotas.morph.open(thresholded_1)
        mahotas.morph.dilate(thresholded_2, mahotas.morph.get_structuring_elem(thresholded_2, 9), out=thresholded_1)
        array[thresholded_1 == 0] = 0
        return array

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._internal.get_image_size_zyx()

    def uncached(self) -> "ImageLoader":
        return BackgroundSupressingImageLoader(self._internal.uncached())

    def first_time_point_number(self) -> Optional[int]:
        return self._internal.first_time_point_number()

    def last_time_point_number(self) -> Optional[int]:
        return self._internal.last_time_point_number()

    def get_channels(self) -> List[ImageChannel]:
        return self._internal.get_channels()

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._internal.serialize_to_config()
