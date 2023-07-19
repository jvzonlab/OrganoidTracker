import numpy
from numpy import ndarray


def image_to_8bit(image: ndarray):
    """Converts and scales the image to an 8-bit image, such that the highest value in the image becomes 255. So even if the image
    is already an 8-bit image, it can still get rescaled.

    Note that this method returns a copy and does not modify the original image."""
    image = image.astype(numpy.float32)
    image += image.min()  # Prevent negative values
    image = image / float(image.max()) * 255.0  # Scale from 0 to 255
    return image.astype(numpy.uint8)  # Convert to 8bit


def ensure_8bit(image: ndarray):
    """Converts an image to uint8 if it isn't already."""
    if image.dtype == numpy.uint8:
        return image
    return image_to_8bit(image)


def add_and_return_8bit(a: ndarray, b: ndarray) -> ndarray:
    """Adds the two arrays. First, the arrays are scaled to 8bit if they aren't already, and then they are added without
    overflow issues: 240 + 80 is capped at 255."""
    a = ensure_8bit(a)
    b = ensure_8bit(b)

    # https://stackoverflow.com/questions/29611185/avoid-overflow-when-adding-numpy-arrays
    b = 255 - b  # old b is gone shortly after new array is created
    numpy.putmask(a, b < a, b)  # a temp bool array here, then it's gone
    a += 255 - b  # a temp array here, then it's gone
    return a
