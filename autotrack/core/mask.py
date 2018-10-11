from abc import ABC
from typing import Optional

import numpy
from numpy import ndarray

from autotrack.core.bounding_box import BoundingBox


class OutsideImageError(Exception):
    """Thrown when a mask falls completely outside an image."""
    pass


class Mask(ABC):
    """Class used for drawing and applying masks."""

    @property
    def offset_x(self):
        raise NotImplementedError()

    @property
    def offset_y(self):
        raise NotImplementedError()

    @property
    def offset_z(self):
        raise NotImplementedError()

    def set_bounds(self, box: BoundingBox):
        """Shrinks the bounding box to the given box. In this way, smaller arrays for the mask can be
        allocated.  Setting a bounding box twice or setting it after get_mask_array
        has been called is not allowed."""
        self.set_bounds_exact(box.min_x, box.min_y, box.min_z, box.max_x, box.max_y, box.max_z)

    def set_bounds_around(self, x: float, y: float, z: float, padding_x: float, padding_y: float, padding_z: float):
        """Sets the bounding box of the shape that is going to be drawn. In this way, smaller arrays for the mask can be
        allocated. Setting a bounding box twice or setting it after get_mask_array has been called is not allowed.

        If all three paddings are set to 0, an image of exactly 1 pixel will remain."""
        self.set_bounds_exact(x - padding_x, y - padding_y, z - padding_z,
                              x + padding_x + 1, y + padding_y + 1, z + padding_z + 1)

    def set_bounds_exact(self, min_x: float, min_y: float, min_z: float, max_x: float, max_y: float, max_z: float):
        """Sets the bounding box of the shape that is going to be drawn. In this way, smaller arrays for the mask can be
        allocated. Setting a bounding box twice or setting it after get_mask_array has been called is not allowed."""
        raise NotImplementedError()

    def get_mask_array(self) -> ndarray:
        """Gets a 3D array to draw the mask on. Make sure to set appropriate bounds for this array first. Note that to
        safe memory the coords of this array are shifted by (offset_x, offset_y, offset_z) compared to the original
        image: in this way the array can be smaller."""
        raise NotImplementedError()

    def create_masked_and_normalized_image(self, image_stack: ndarray):
        """Creates a copy of the given image with all pixels not in the mask set to 0 (black)."""
        raise NotImplementedError()


class _SingleMask(Mask):
    _offset_x: int = 0
    _offset_y: int = 0
    _offset_z: int = 0
    _max_x: int
    _max_y: int
    _max_z: int

    _mask: Optional[ndarray] = None

    def __init__(self, max_x: int, max_y: int, max_z: int):
        self._max_x = max_x
        self._max_y = max_y
        self._max_z = max_z

    @property
    def offset_x(self):
        return self._offset_x

    @property
    def offset_y(self):
        return self._offset_y

    @property
    def offset_z(self):
        return self._offset_z

    def set_bounds_exact(self, min_x: float, min_y: float, min_z: float, max_x: float, max_y: float, max_z: float):
        if self._mask is not None:
            raise ValueError("Mask already created, cannot resize anymore")
        self._offset_x = max(self._offset_x, int(min_x))
        self._offset_y = max(self._offset_y, int(min_y))
        self._offset_z = max(self._offset_z, int(min_z))
        self._max_x = min(self._max_x, int(max_x))
        self._max_y = min(self._max_y, int(max_y))
        self._max_z = min(self._max_z, int(max_z))

    def get_mask_array(self):
        if self._mask is None:
            size_x = self._max_x - self._offset_x
            size_y = self._max_y - self._offset_y
            size_z = self._max_z - self._offset_z
            if size_x <= 0 or size_y <= 0 or size_z <= 0:
                raise OutsideImageError()  # Attempting to create mask of size 0
            self._mask = numpy.zeros((size_z, size_y, size_x), dtype=numpy.uint8)
        return self._mask

    def create_masked_and_normalized_image(self, image_stack: ndarray) -> ndarray:
        """Create normalized subimage. Throws OutsideImageError when the mask is fully outside the image."""
        image_for_masking = image_stack[self._offset_z:self._max_z,
                            self._offset_y:self._max_y,
                            self._offset_x:self._max_x].astype(dtype=numpy.float32)
        try:
            image_for_masking /= image_for_masking.max()
        except ValueError:
            raise OutsideImageError()

        # Crop mask to same size as subimage
        mask = self._mask[0:image_for_masking.shape[0], 0:image_for_masking.shape[1], 0:image_for_masking.shape[2]]

        # Apply mask
        image_for_masking[mask == 0] = numpy.NAN
        return image_for_masking


def create_mask_for(image: ndarray) -> Mask:
    """Creates a mask that will never expand beyond the size of the given image."""
    return _SingleMask(image.shape[2], image.shape[1], image.shape[0])
