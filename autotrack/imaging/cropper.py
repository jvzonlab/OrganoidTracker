from numpy import ndarray


def crop_2d(image: ndarray, x_start: int, y_start: int, z: int, output: ndarray):
    """Gets a cropped 2d image from the original image. Any pixel outside the original image is ignored."""
    if x_start >= image.shape[2]:
        return  # We're completely outside the image, nothing to do
    if y_start >= image.shape[1]:
        return  # We're completely outside the image, nothing to do
    if z < 0 or z >= image.shape[0]:
        return  # We're completely outside the image, nothing to do

    # Calculate input image bounds
    x_size = output.shape[1]
    y_size = output.shape[0]

    if x_start + x_size <= 0 or y_start + y_size <= 0:
        return  # We're completely outside the image, nothing to do
    if x_start + x_size > image.shape[2]:
        x_size = image.shape[2] - x_start  # Partly outside image, reduce size
    if y_start + y_size > image.shape[1]:
        y_size = image.shape[1] - y_start  # Partly outside image, reduce size

    # Correct this if we're requesting pixels outside the input image
    output_x_offset = 0
    output_y_offset = 0
    if x_start < 0:
        output_x_offset = -x_start
        x_start = 0
        x_size -= output_x_offset
    if y_start < 0:
        output_y_offset = -y_start
        y_start = 0
        y_size -= output_y_offset

    output[output_y_offset:output_y_offset + y_size, output_x_offset:output_x_offset + x_size] \
        = image[z, y_start:y_start + y_size, x_start:x_start + x_size]
