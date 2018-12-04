from autotrack.core.image_loader import ImageLoader
from autotrack.core.particles import Particle
from numpy import ndarray


class MaximumIntensityProjector:
    _image_loader: ImageLoader
    _num_layers: int

    def __init__(self, image_loader: ImageLoader, num_layers: int):
        """Creates a maximum intensity projector, which can be used to create those projections of particles."""
        self._image_loader = image_loader
        self._num_layers = num_layers

    def create_projection(self, particle: Particle, out_array: ndarray):
        """Creates a maximum intensity projection and stores it in the out_array (a 2D array). The x/y size of the
        projection depends on the size of out_array."""
        width = out_array.shape[1]
        height = out_array.shape[0]
        x_start = max(0, int(particle.x - width // 2))
        y_start = max(0, int(particle.y - height // 2))
        z_start = max(0, int(particle.z - self._num_layers // 2))

        image = self._image_loader.get_image_stack(particle.time_point())
        if image is None:
            print("No image for time point", particle.time_point(), "found")
            return
        width = min(width, image.shape[2] - x_start)
        height = min(height, image.shape[1] - y_start)
        num_layers = min(self._num_layers, image.shape[0] - z_start)

        sub_image: ndarray = image[z_start:z_start + num_layers, y_start:y_start + height, x_start:x_start + width]
        sub_image.max(axis=0, out=out_array[:height, :width])
