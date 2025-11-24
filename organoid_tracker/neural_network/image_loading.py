from typing import Dict, Optional, Tuple

import numpy

from organoid_tracker.core import TimePoint
from organoid_tracker.core.images import Image
from organoid_tracker.imaging import cropper


def fill_none_images_with_copies(full_images: Dict[TimePoint, Optional[Image]]):
    """Fills any None values in the array with the array of the closest time point that is available.

    (At the start or end of a movie, some time points may be missing. In that case, it's best to just copy the nearest
    available image.)

    Raises a ValueError if no images are available at all.
    """
    time_points = list(full_images.keys())
    for time_point in time_points:
        if full_images[time_point] is not None:
            continue
        # Find nearest available image
        offset = 1
        while True:
            time_point_before = TimePoint(time_point.time_point_number() - offset)
            image_before = full_images.get(time_point_before)
            if image_before is not None:
                full_images[time_point] = image_before
                break

            time_point_after = TimePoint(time_point.time_point_number() + offset)
            image_after = full_images.get(time_point_after)
            if image_after is not None:
                full_images[time_point] = image_after
                break

            offset += 1
            if offset > len(time_points):
                raise ValueError("No images available to fill missing time points.")


def extract_patch_array(full_images: Dict[TimePoint, Image], start_zyx: Tuple[int, int, int],
                   patch_shape_zyx: Tuple[int, int, int]) -> numpy.ndarray:
    """Extracts a patch from the full images. Coordinates are assumed to be in position coordinates. Offsets of the
    images are taken into account.

    full_images must contain a continuous set of time points, but doesn't have to be in order.
    """

    min_time_point = min(full_images.keys())
    output_array = numpy.zeros((patch_shape_zyx[0], patch_shape_zyx[1], patch_shape_zyx[2], len(full_images)), dtype=numpy.float32)

    for time_point_dt, image in full_images.items():
        dt_index = time_point_dt.time_point_number() - min_time_point.time_point_number()
        offset = image.offset
        x_start = int(start_zyx[2] - offset.x)
        y_start = int(start_zyx[1] - offset.y)
        z_start = int(start_zyx[0] - offset.z)

        cropper.crop_3d(image.array, x_start, y_start, z_start, output_array[:, :, :, dt_index])

    return output_array