from numpy import ndarray


class ImageEdgeError(LookupError):
    pass


def get_square(image: ndarray, center_x: float, center_y: float, r: int) -> ndarray:
    """Imagine a square centered around (center_x, center_y) with radius r. This method returns all intensities in
    that square as a 2D numpy array. The square has a size of 2r by 2r. Intensities are scaled from 0 to 1, where
    1 is the highest intensity found in the complete image (not this square)
    """
    # Top and bottom rows
    y_top = int(center_y) - r
    y_bottom = int(center_y) + r
    x_left = int(center_x) - r
    x_right = int(center_x) + r

    if x_left < 0 or x_right > image.shape[1] or y_top < 0 or y_bottom > image.shape[0]:
        raise ImageEdgeError("Out of bounds: " + str((center_x, center_y, r)))

    return image[y_top : y_bottom, x_left : x_right] / image.max()
