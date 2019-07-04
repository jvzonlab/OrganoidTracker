"""For slicing big images into smaller images."""
from typing import Tuple, Iterable

import numpy
from numpy import ndarray

from autotrack.imaging import cropper


class Slicer3d:
    """Class to slice a big image into smaller slices, and to put those smaller slices back into a big image. This is
    useful if you want to process a big image in smaller parts, for example to save RAM.

    The slicer supports padding: around an area of interest, you can have some more pixels from the big image. This
    makes it easier to handle the edges between the slices.
    """

    _start: Tuple[int, int, int]  # Start position of the slice, inclusive
    _end: Tuple[int, int, int]  # End position of the slice, exclusive
    _area_of_interest_start: Tuple[int, int, int]
    _area_of_interest_end: Tuple[int, int, int]

    def __init__(self, start: Tuple[int, int, int], end: Tuple[int, int, int],
                 area_of_interest_start: Tuple[int, int, int], area_of_interest_end: Tuple[int, int, int]):
        # Some sanity checks
        if start[0] < 0 or start[1] < 0 or start[2] < 0:
            raise ValueError(f"start cannot be negative; was {start}")
        if area_of_interest_start[0] < start[0] or area_of_interest_start[1] < start[1]\
                or area_of_interest_start[2] < start[2]:
            raise ValueError(f"area_of_interest cannot start before the start; start={start},"
                             f" area_of_interest_start={area_of_interest_start}")
        if area_of_interest_end[0] > end[0] or area_of_interest_end[1] > end[1] or area_of_interest_end[2] > end[2]:
            raise ValueError(f"area_of_interest cannot end after the end; end={end},"
                             f" area_of_interest_end={area_of_interest_end}")

        self._start = start
        self._end = end
        self._area_of_interest_start = area_of_interest_start
        self._area_of_interest_end = area_of_interest_end

    def slice(self, image_3d: ndarray) -> ndarray:
        if self._end[0] > image_3d.shape[0] or self._end[1] > image_3d.shape[1] or self._end[2] > image_3d.shape[2]:
            # Need to allocate bigger image to make sure slice is of the right size
            # Should only happen if the requested slice size is bigger than the original image in at least one dimension
            new_image = numpy.zeros((self._end[0] - self._start[0], self._end[1] - self._start[1], self._end[2] - self._start[2]))
            cropper.crop_3d(image_3d, self._start[2], self._start[1], self._start[0], new_image)
            return new_image
        return image_3d[self._start[0]:self._end[0], self._start[1]:self._end[1], self._start[2]:self._end[2]]

    def place_slice_in_volume(self, slice: ndarray, volume: ndarray):
        """Used to stitch an image back together. Places a slice (more precise: the area of interest of the slice)
        created using self.slice(..) back in an array of the same size as the array given to self.slice(..)."""
        volume[self._area_of_interest_start[0]:self._area_of_interest_end[0],
               self._area_of_interest_start[1]:self._area_of_interest_end[1],
               self._area_of_interest_start[2]:self._area_of_interest_end[2]] = slice[
                        self._area_of_interest_start[0] - self._start[0]:self._area_of_interest_end[0] - self._start[0],
                        self._area_of_interest_start[1] - self._start[1]:self._area_of_interest_end[1] - self._start[1],
                        self._area_of_interest_start[2] - self._start[2]:self._area_of_interest_end[2] - self._start[2]]

    def __repr__(self) -> str:
        return f"Slicer3d(start={self._start}, end={self._end}, area_of_interest_start={self._area_of_interest_start}, area_of_interest_end={self._area_of_interest_end})"

    def __eq__(self, other):
        if not isinstance(other, Slicer3d):
            return False
        return other._start == self._start and other._end == self._end\
               and other._area_of_interest_start == self._area_of_interest_start\
               and other._area_of_interest_end == self._area_of_interest_end


def get_slices(volume_zyx: Tuple[int, int, int], image_part_size: Tuple[int, int, int],
                image_part_margin: Tuple[int, int, int]) -> Iterable[Slicer3d]:
    """Slices a big image into smaller sub-images. The sub-images will have always have a size of
    `image_part_size + 2 * image_part_margin` and will overlap with each other by at least image_part_margin pixels.
    All tuples are formatted ZYX."""
    start_z = 0
    while start_z < volume_zyx[0]:
        start_y = 0
        while start_y < volume_zyx[1]:
            start_x = 0
            while start_x < volume_zyx[2]:
                # Create start/end coords that don't exceed the bounds of the whole image
                start_x_here = max(0, start_x - image_part_margin[2])
                start_y_here = max(0, start_y - image_part_margin[1])
                start_z_here = max(0, start_z - image_part_margin[0])
                end_x_here = start_x_here + image_part_size[2] + 2 * image_part_margin[2]
                end_y_here = start_y_here + image_part_size[1] + 2 * image_part_margin[1]
                end_z_here = start_z_here + image_part_size[0] + 2 * image_part_margin[0]
                if end_x_here > volume_zyx[2]:
                    start_x_here = max(0, start_x_here - (end_x_here - volume_zyx[2]))
                    end_x_here = start_x_here + image_part_size[2] + 2 * image_part_margin[2]
                if end_y_here > volume_zyx[1]:
                    start_y_here = max(0, start_y_here - (end_y_here - volume_zyx[1]))
                    end_y_here = start_y_here + image_part_size[1] + 2 * image_part_margin[1]
                if end_z_here > volume_zyx[0]:
                    start_z_here = max(0, start_z_here - (end_z_here - volume_zyx[0]))
                    end_z_here = start_z_here + image_part_size[0] + 2 * image_part_margin[0]
                end_x = min(start_x + image_part_size[2], end_x_here)  # Don't let the area of interest extend beyond
                end_y = min(start_y + image_part_size[1], end_y_here)  # the area of the slice
                end_z = min(start_z + image_part_size[0], end_z_here)

                yield Slicer3d((start_z_here, start_y_here, start_x_here), (end_z_here, end_y_here, end_x_here),
                               (start_z, start_y, start_x), (end_z, end_y, end_x))
                start_x += image_part_size[2]
            start_y += image_part_size[1]
        start_z += image_part_size[0]
