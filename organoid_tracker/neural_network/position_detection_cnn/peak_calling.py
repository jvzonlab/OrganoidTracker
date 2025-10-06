from typing import Tuple


import math

import numpy
from numpy import ndarray

from organoid_tracker.core import bounding_box
from organoid_tracker.core.mask import Mask
from organoid_tracker.core.resolution import ImageResolution


# Helps peak calling
def reconstruct_volume(multi_im: ndarray, mid_layers_nb: int) -> Tuple[ndarray, int]:
    """Reconstructs a volume so that the xy scale is roughly equal to the z scale. Returns the used scale."""
    # Make sure that
    if mid_layers_nb < 0:
        raise ValueError("negative number of mid layers")
    if mid_layers_nb == 0:
        return multi_im, 1  # No need to reconstruct anything

    out_img = numpy.zeros(
        (int(len(multi_im) + mid_layers_nb * (len(multi_im) - 1) + 2 * mid_layers_nb),) + multi_im[0].shape, dtype=
        multi_im[0].dtype)

    layer_index = mid_layers_nb + 1
    orig_index = []

    for i in range(len(multi_im) - 1):

        for layer in range(mid_layers_nb + 1):
            t = float(layer) / (mid_layers_nb + 1)
            interpolate = ((1 - t) * (multi_im[i]).astype(float) + t * (multi_im[i + 1]).astype(float))

            out_img[layer_index] = interpolate

            if t == 0:
                orig_index.append(layer_index)
            layer_index += 1

    return out_img, mid_layers_nb + 1



def create_prediction_mask(radius_x_px_float: float, resolution: ImageResolution) -> ndarray:
    """Creates a mask that is spherical in micrometers. If the resolution is not the same in the x, y and z directions,
    this sphere will appear as a spheroid in the images.

    For use with the footprint param in scikit-image peak_local_max."""
    radius_x_px = int(math.ceil(radius_x_px_float))
    radius_y_px = int(math.ceil(radius_x_px_float) * resolution.pixel_size_x_um / resolution.pixel_size_y_um)
    radius_z_px = int(math.ceil(radius_x_px_float * resolution.pixel_size_x_um / resolution.pixel_size_z_um))
    mask = Mask(bounding_box.ONE.expanded(radius_x_px, radius_y_px, radius_z_px))

    # Evaluate the spheroid function to draw it
    mask.add_from_function(lambda x, y, z:
                           x ** 2 / radius_x_px ** 2 + y ** 2 / radius_y_px ** 2 + z ** 2 / radius_z_px ** 2 <= 1)

    return mask.get_mask_array()
