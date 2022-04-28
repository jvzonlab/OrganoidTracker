"""Used to add the pixels of two images to each other, with various offsets."""
from typing import Tuple, Iterable

import numpy
from numpy import ndarray

from organoid_tracker.core.images import Image
from organoid_tracker.imaging import cropper


def add_images(drawing_board: Image, stamp: Image):
    """Adds the contents of the second image to the first."""
    min_x = max(drawing_board.min_x, stamp.min_x)
    min_y = max(drawing_board.min_y, stamp.min_y)
    min_z = max(drawing_board.min_z, stamp.min_z)
    max_x = min(drawing_board.limit_x, stamp.limit_x)
    max_y = min(drawing_board.limit_y, stamp.limit_y)
    max_z = min(drawing_board.limit_z, stamp.limit_z)

    if min_x >= max_x or min_y >= max_y or min_z >= max_z:
        return  # No overlap in between the two images

    offset_x, offset_y, offset_z = int(stamp.offset.x), int(stamp.offset.y), int(stamp.offset.z)
    from_stamp = stamp.array[min_z - offset_z:max_z - offset_z, min_y - offset_y:max_y-offset_y, min_x-offset_x:max_x-offset_x]

    offset_x, offset_y, offset_z = int(drawing_board.offset.x), int(drawing_board.offset.y), int(drawing_board.offset.z)
    drawing_board.array[min_z - offset_z:max_z - offset_z, min_y - offset_y:max_y-offset_y, min_x-offset_x:max_x-offset_x] += from_stamp
