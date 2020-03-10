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

    image_2d = _load_2d_image(file_name)
    if image_2d is None:
        return None
    return image_2d[numpy.newaxis, ...]  # Add a z-axis


def read_image_2d(file_name: str, image_z: int) -> Optional[ndarray]:
    """Reads an image file that represents a single time point. Container files (like LIF or ND) are not supported. A
    2D grayscale image is returned. In case the original image was 2D, the image is only returned if image_z == 0.
    Colored images are converted to grayscale. Returns None if the file does not exist, does not have an image at the
    given image_z or if the file cannot be read."""
    if image_z < 0:
        return None  # Will never return a value
    if not os.path.exists(file_name):
        return None

    file_name_lower = file_name.lower()

    # Load from TIFF
    if file_name_lower.endswith(".tif") or file_name_lower.endswith(".tiff"):
        return _load_2d_image_from_tiff(file_name, image_z)

    # Load from a 2D file (only if z == 0, at other z no image is available)
    if image_z == 0:
        return _load_2d_image(file_name)
    return None


def _load_tiff(file_name: str) -> Optional[ndarray]:
    """For TIFF files."""
    import tifffile
    with tifffile.TiffFile(file_name) as f:
        # noinspection PyTypeChecker
        array = numpy.squeeze(f.asarray(maxworkers=None))
        # ^ maxworkers=None makes image loader work on half of all cores
        if len(f.pages) == 1:
            # If we have a single page, make it a 3D image anyways
            array = array[numpy.newaxis, ...]
        if len(array.shape) == 4:
            # 4 dimensions, so we have a colored image. Convert to grayscale
            array = _to_grayscale(array)
        if len(array.shape) == 3:
            # We should now have a 3D image
            return array
        return None  # Weird TIFF file that cannot be read


def _load_2d_image(file_name: str) -> Optional[ndarray]:
    """For simple 2d images that may be colored, like PNG, JPG and GIF."""
    try:
        image_2d = matplotlib.image.imread(file_name)
    except ValueError:
        return None
    else:
        if len(image_2d.shape) == 3:
            # 3 dimensions, so we have a colored image. Convert to grayscale
            image_2d = _to_grayscale(image_2d)
        if len(image_2d.shape) != 2:
            raise ValueError(f"Got an image of unexpected shape: {image_2d.shape}. This is bug!")
        return image_2d


def _load_2d_image_from_tiff(file_name: str, image_z: int) -> Optional[ndarray]:
    """For TIFF files."""
    import tifffile
    with tifffile.TiffFile(file_name) as f:
        if image_z < 0 or image_z >= len(f.pages):
            return None

        # noinspection PyTypeChecker
        array = numpy.squeeze(f.asarray(maxworkers=None, key=image_z))
        # ^ maxworkers=None makes image loader work on half of all cores
        if len(array.shape) == 3:
            # 3 dimensions in a single page, so we have a colored image. Convert to grayscale
            array = _to_grayscale(array)
        if len(array.shape) == 2:
            # We should now have a 2D image
            return array
        return None  # Weird TIFF file that cannot be read


def _to_grayscale(array: ndarray) -> ndarray:
    """Converts an N-dimensional image to grayscale. This removes the last dimension of the array
     (assumed to be color)."""
    return numpy.dot(array[..., :3], [0.299, 0.587, 0.114])






