from numpy import ndarray
import numpy
import matplotlib.cm
from PIL import Image

from ai_track.core.images import Images


def create_image(image: ndarray, color_map_name: str = "Spectral") -> ndarray:
    """Creates a 2D image (float32, [y, x, RGBA]) by giving each xy later in the 3D image another color."""
    color_map = matplotlib.cm.get_cmap(color_map_name)

    black_image = numpy.zeros((image.shape[1], image.shape[2], 4), dtype=numpy.uint8)
    black_image[:, :, 3] = 255  # Set alpha to opaque
    color_image_pil = Image.fromarray(black_image)

    max_z = image.shape[0] - 1
    max_intensity = image.max()

    slice_buffer = numpy.zeros((image.shape[1], image.shape[2], 4), dtype=numpy.float32)  # 2D RGBA
    slice_buffer_uint8 = numpy.zeros((image.shape[1], image.shape[2], 4), dtype=numpy.uint8)  # 2D RGBA

    for z in range(max_z, -1, -1):
        image_slice = image[z]

        # Make the temporary buffer colored
        color = color_map(z / max_z)
        slice_buffer[:, :, 0] = color[0] * 255
        slice_buffer[:, :, 1] = color[1] * 255
        slice_buffer[:, :, 2] = color[2] * 255

        # Set the alpha layer to the image slice
        slice_buffer[:, :, 3] = image_slice
        slice_buffer[:, :, 3] /= max_intensity
        slice_buffer[:, :, 3] **= 2  # To suppress noise
        slice_buffer[:, :, 3] *= 255

        # Add to existing image
        slice_buffer_uint8[...] = slice_buffer
        slice_buffer_pil = Image.fromarray(slice_buffer_uint8)
        if color_image_pil is None:
            color_image_pil = slice_buffer_pil
        else:
            color_image_pil = Image.alpha_composite(color_image_pil, slice_buffer_pil)

    color_image = numpy.asarray(color_image_pil, dtype=numpy.float32)
    color_image /= 255  # Scale to 0-1
    return color_image


def create_movie(images: Images) -> ndarray:
    """Creates a movie of 2D images (float32, [time, y, x, RGB]) similar to create_image, but for every time point."""
    last_time_point = images.image_loader().last_time_point_number()
    first_time_point = images.image_loader().first_time_point_number()

    image_count = 0 if last_time_point is None else last_time_point - first_time_point + 1
    if image_count == 0:
        raise ValueError("No images found")
    image_shape = images.image_loader().get_image_size_zyx()
    total_image = numpy.zeros((image_count, image_shape[1], image_shape[2], 3), dtype=numpy.uint8)

    i = 0
    for time_point in images.time_points():
        print(f"Working on time point {time_point.time_point_number()}...")
        image = images.get_image_stack(time_point)
        colored_image = create_image(image)
        total_image[i] = (colored_image[:, :, 0:3] * 255).astype(numpy.uint8)

        i += 1

    return total_image
