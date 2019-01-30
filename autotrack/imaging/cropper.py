from numpy import ndarray

from autotrack.core.images import Image


def crop_2d(image: Image, x_start: int, y_start: int, z: int, output: ndarray):
    """Gets a cropped 2d image from the original 3d image. The start of the cropped image is defined by (x_start,
    y_start) and the size by the size of the output image. If the area to crop falls partly outside the original image,
    then those pixels are ignored and the original pixels in output will remain. Therefore, the output array must be an
    array of only zeros."""
    if x_start >= image.limit_x:
        return  # We're completely outside the image, nothing to do
    if y_start >= image.limit_y:
        return  # We're completely outside the image, nothing to do
    if z < 0 or z >= image.limit_z:
        return  # We're completely outside the image, nothing to do

    # Calculate input image bounds
    x_size = output.shape[1]
    y_size = output.shape[0]

    if x_start + x_size <= image.min_x or y_start + y_size <= image.min_y:
        return  # We're completely outside the image, nothing to do
    if x_start + x_size > image.limit_x:
        x_size = image.limit_x - x_start  # Partly outside image, reduce size
    if y_start + y_size > image.limit_y:
        y_size = image.limit_y - y_start  # Partly outside image, reduce size

    # Correct this if we're requesting pixels outside the input image
    output_x_offset = 0
    output_y_offset = 0
    if x_start < image.min_x:
        output_x_offset = image.min_x - x_start
        x_start = image.min_x
        x_size -= output_x_offset
    if y_start < image.min_y:
        output_y_offset = image.min_y - y_start
        y_start = image.min_y
        y_size -= output_y_offset

    output[output_y_offset:output_y_offset + y_size, output_x_offset:output_x_offset + x_size] \
        = image.array[z - image.min_z, y_start - image.min_y:y_start - image.min_y + y_size, x_start - image.min_x:x_start - image.min_x + x_size]


def crop_3d(image: ndarray, x_start: int, y_start: int, z_start: int, output: ndarray):
    """Similar to crop_2d, except that all layers are now copied instead of only a single layer. Output must now be a
    3D image with the same number of layers as the input image."""
    if x_start >= image.shape[2]:
        return  # We're completely outside the image, nothing to do
    if y_start >= image.shape[1]:
        return  # We're completely outside the image, nothing to do
    if z_start >= image.shape[0]:
        return  # We're completely outside the image, nothing to do

    # Calculate input image bounds
    x_size = output.shape[2]
    y_size = output.shape[1]
    z_size = output.shape[0]

    if x_start + x_size <= 0 or y_start + y_size <= 0:
        return  # We're completely outside the image, nothing to do
    if x_start + x_size > image.shape[2]:
        x_size = image.shape[2] - x_start  # Partly outside image, reduce size
    if y_start + y_size > image.shape[1]:
        y_size = image.shape[1] - y_start  # Partly outside image, reduce size
    if z_start + z_size > image.shape[0]:
        z_size = image.shape[0] - z_start  # Partly outside image, reduce size

    # Correct this if we're requesting pixels outside the input image
    output_x_offset = 0
    output_y_offset = 0
    output_z_offset = 0
    if x_start < 0:
        output_x_offset = -x_start
        x_start = 0
        x_size -= output_x_offset
    if y_start < 0:
        output_y_offset = -y_start
        y_start = 0
        y_size -= output_y_offset
    if z_start < 0:
        output_z_offset = -z_start
        z_start = 0
        z_size -= output_z_offset

    output[output_z_offset:output_z_offset + z_size, output_y_offset:output_y_offset + y_size, output_x_offset:output_x_offset + x_size] \
        = image[z_start:z_start + z_size, y_start:y_start + y_size, x_start:x_start + x_size]
