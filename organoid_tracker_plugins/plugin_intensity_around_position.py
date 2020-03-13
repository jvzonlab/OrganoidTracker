from math import ceil

from matplotlib import pyplot
from tifffile import tifffile

from organoid_tracker.core import bounding_box
from organoid_tracker.core.bounding_box import BoundingBox
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import Image
from organoid_tracker.core.mask import Mask
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui.window import Window


def _calculate(experiment: Experiment, channel_1: ImageChannel, channel_2: ImageChannel, radius_um: float):
    mask = _create_spherical_mask(radius_um, experiment.images.resolution())

    for time_point in experiment.positions.time_points():
        print("Working on time point", time_point.time_point_number())
        image_1 = experiment.images.get_image(time_point, channel_1)
        image_2 = experiment.images.get_image(time_point, channel_2)
        for position in experiment.positions.of_time_point(time_point):
            intensity_1 = _get_averaged_intensity_for(image_1, mask, position)
            intensity_2 = _get_averaged_intensity_for(image_2, mask, position)
            print(intensity_1 / intensity_2)


def _create_spherical_mask(radius_um: float, resolution: ImageResolution) -> Mask:
    """Creates a mask that is spherical in micrometers. If the resolution is not the same in the x, y and z directions,
    this sphere will appear as a spheroid in the images."""
    radius_x_px = ceil(radius_um / resolution.pixel_size_x_um)
    radius_y_px = ceil(radius_um / resolution.pixel_size_y_um)
    radius_z_px = ceil(radius_um / resolution.pixel_size_z_um)
    mask: Mask = Mask(bounding_box.ONE.expanded(radius_x_px, radius_y_px, radius_z_px))

    # Evaluate the spheroid function to draw it
    mask.add_from_function(lambda x, y, z: \
            x ** 2 / radius_x_px ** 2 + y ** 2 / radius_y_px ** 2 + z ** 2 / radius_z_px ** 2 <= 1)

    return mask

def _get_averaged_intensity_for(image: Image, mask: Mask, position: Position) -> float:
    mask.center_around(position)
    masked_image = mask.create_masked_image(image)
    return float(masked_image.sum()) / mask.count_pixels()
