"""Helper for reading the image file of a single time point."""
import os
from typing import Optional

import matplotlib.image
import numpy
from numpy import ndarray


def read_image_3d(file_name: str) -> Optional[ndarray]:
    """Reads an image file that represents a single time point. Container files (like LIF or ND) are not supported. A
    3D grayscale image is returned. In case the original image was 2D, a 3D image with one xy plane is returned. Colored
    images are converted to grayscale. Returns None if the file does not exist or cannot be read."""
    if not os.path.exists(file_name):
        return None

    file_name_lower = file_name.lower()
    if file_name_lower.endswith(".tif") or file_name_lower.endswith(".tiff"):
        return _load_tiff(file_name)
    return _load_2d_image(file_name)


def _load_tiff(file_name: str) -> Optional[ndarray]:
    """For TIFF files."""
    import tifffile
    with tifffile.TiffFile(file_name, movie=True) as f:
        # noinspection PyTypeChecker
        array = numpy.squeeze(f.asarray(maxworkers=None))
        # ^ maxworkers=None makes image loader work on half of all cores
        if len(f.pages) == 1:
            # If we have a single page, make it a 3D image anyways
            array = array[numpy.newaxis, ...]
        if len(array.shape) == 4:
            # 4 dimensions, so we have a colored image. Convert to grayscale
            array = numpy.dot(array[..., :3], [0.299, 0.587, 0.114])
        if len(array.shape) == 3:
            # We should now have a 3D image
            return array
        return None  # Weird TIFF file that cannot be read


def _load_2d_image(file_name: str) -> Optional[ndarray]:
    """For simple 2d images that may be colored, like PNG, JPG and GIF"""
    try:
        image_2d = matplotlib.image.imread(file_name)
    except ValueError:
        return None
    else:
        if len(image_2d.shape) == 3:
            # 3 dimensions, so we have a colored image. Convert to grayscale
            image_2d = numpy.dot(image_2d[..., :3], [0.299, 0.587, 0.114])
        if len(image_2d.shape) != 2:
            raise ValueError(f"Got an image of unexpected shape: {image_2d.shape}. This is bug!")
        return image_2d[numpy.newaxis, ...]




