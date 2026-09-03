"""Helper for reading the image file of a single time point."""
import os
from typing import Optional

import matplotlib.image
import numpy
from numpy import ndarray


def read_image_czyx(file_name: str) -> Optional[ndarray]:
    """Reads an image file that represents a single time point. Container files (like LIF or ND) are not supported. A
    3D grayscale image is returned. In case the original image was 2D, a 3D image with one xy plane is returned. Colored
    images are converted to grayscale. Returns None if the file does not exist or cannot be read."""
    if not os.path.exists(file_name):
        return None

    file_name_lower = file_name.lower()
    if file_name_lower.endswith(".tif") or file_name_lower.endswith(".tiff"):
        return _load_tiff_czyx(file_name)

    image_2d = _load_2d_cyx_image(file_name)
    if image_2d is None:
        return None
    return image_2d[:, numpy.newaxis, :, :]  # Add a z-axis


def read_image_cyx(file_name: str, image_z: int) -> Optional[ndarray]:
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
        image = _load_2d_cyx_image_from_tiff(file_name, image_z)
        if image is not None:
            return image
        return None

    # Load from a 2D file (only if z == 0, at other z no image is available)
    if image_z == 0:
        return _load_2d_cyx_image(file_name)
    return None


def _load_tiff_czyx(file_name: str) -> Optional[ndarray]:
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
            # 4 dimensions, so we have a colored image
            if not _rgb_are_different(array[array.shape[0] // 2, ...]):
                # If the color are the same, we have a grayscale image that is stored as a colored image
                array = array[:, :, :, 2]  # Take only the blue channel
                return array[numpy.newaxis, ...]  # Add a color axis

            # Just move the color axis to the front, so we have (C, Z, Y, X) instead of (Z, Y, X, C)
            return numpy.moveaxis(array, 3, 0)
        if len(array.shape) == 3:
            # We have a 3D grayscale image, so we can return it with the color axis added
            return array[numpy.newaxis, ...]
        return None  # Weird TIFF file that cannot be read



def _rgb_are_different(example_image_yxc: numpy.ndarray) -> bool:
    """Checks if the red and blue channel of an (Y, X, 3) image are different.
    Returns True if they are different, False if they are the same."""

    if example_image_yxc.shape[0] > 100:
        # Remove bottom pixels, as those may contain burned-in metadata
        example_image_yxc = example_image_yxc[:-50, :, :]

    return ((example_image_yxc[:, :, 0] != example_image_yxc[:, :, 2]).any()
            or (example_image_yxc[:, :, 1] != example_image_yxc[:, :, 2]).any())


def _load_2d_cyx_image(file_name: str) -> Optional[ndarray]:
    """For simple 2d images that may be colored, like PNG, JPG and GIF."""
    try:
        image_2d = matplotlib.image.imread(file_name)
    except ValueError:
        return None
    else:
        if len(image_2d.shape) == 3:
            # 3 dimensions, so we have a colored image
            if not _rgb_are_different(image_2d):
                # If the color are the same, we have a grayscale image that is stored as a colored image
                image_2d = image_2d[:, :, 2]  # Take only the blue channel
                return image_2d[numpy.newaxis, ...]  # Add a color axis
            else:
                # Move the color axis to the front, so we have (C, Y, X) instead of (Y, X, C)
                image_2d = numpy.moveaxis(image_2d, 2, 0)
                return image_2d

        if len(image_2d.shape) == 2:
            return image_2d[numpy.newaxis, ...]  # Add a color axis

    raise ValueError(f"Got an image of unexpected shape: {image_2d.shape}. This is bug!")

def _load_2d_cyx_image_from_tiff(file_name: str, image_z: int) -> Optional[ndarray]:
    """For TIFF files."""
    import tifffile
    with tifffile.TiffFile(file_name) as f:
        if image_z < 0 or image_z >= len(f.pages):
            return None

        # noinspection PyTypeChecker
        array = numpy.squeeze(f.asarray(maxworkers=None, key=image_z))
        # ^ maxworkers=None makes image loader work on half of all cores
        if len(array.shape) == 3:
            # 3 dimensions in a single page, so we have a colored image
            if not _rgb_are_different(array):
                # If the color are the same, we have a grayscale image that is stored as a colored image
                array = array[:, :, 2]  # Take only the blue channel
                return array[numpy.newaxis, ...]  # Add a color axis

            # Move the color axis to the front, so we have (C, Y, X) instead of (Y, X, C)
            return numpy.moveaxis(array, 2, 0)
        if len(array.shape) == 2:
            # 2 dimensions in a single page, so we have a grayscale image. Add a color axis
            return array[numpy.newaxis, ...]
        return None  # Weird TIFF file that cannot be read


def write_image_czyx(file_name: str, image: ndarray):
    """Writes a 3D image to a file. The image must be in the format (C, Z, Y, X). TIFF, JPG and PNG are supported."""
    if image.shape[0] not in {1, 3, 4}:
        raise ValueError("Only grayscale, RGB or RGBA images are supported.")
    grayscale = image.shape[0] == 1

    # Convert to ZYX(C) format for saving
    if image.shape[0] == 1:
        image = image[0, ...]  # Remove the channel axis for single-channel images
    else:
        image = numpy.moveaxis(image, 0, -1)  # Change to ZYXC format for saving

    file_name_lower = file_name.lower()
    if file_name_lower.endswith(".tif") or file_name_lower.endswith(".tiff"):
        # If we have a TIFF image, save it as such
        import tifffile
        tifffile.imwrite(file_name, image, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})
    elif image.shape[0] == 1:
        # Not a TIFF. If we only have a single z-plane, save as 2D image
        if grayscale:
            matplotlib.image.imsave(file_name, image[0], cmap="gray")
        else:
            matplotlib.image.imsave(file_name, image[0])
    else:
        raise ValueError("Only TIFF files are supported for 3D images.")
