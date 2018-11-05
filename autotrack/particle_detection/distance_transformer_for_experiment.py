from pathlib import Path

import numpy
from scipy.ndimage import morphology
import tifffile

from autotrack.core.experiment import Experiment
from autotrack.particle_detection import thresholding
from os import path


def perform_for_experiment(experiment: Experiment, output_folder: str, block_size: int = 1, max_distance: float = 1000):
    sampling = experiment.image_resolution().pixel_size_zyx_um
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for time_point in experiment.time_points():
        print("Working on time point " + str(time_point.time_point_number()) + "...")
        image = thresholding.image_to_8bit(experiment.get_image_stack(time_point))
        threshold = numpy.empty_like(image)

        # Distance transform from threshold
        thresholding.adaptive_threshold(image, threshold, block_size)
        distances = numpy.empty_like(image, dtype=numpy.float64)
        morphology.distance_transform_edt(threshold.max() - threshold, sampling=sampling, distances=distances)
        distances[distances > max_distance] = max_distance
        distances_8bit = thresholding.image_to_8bit(distances)
        tifffile.imsave(path.join(output_folder, "from-threshold-t%03d.tif" % time_point.time_point_number()),
                        distances_8bit)

        # Distance transform from cell positions
        threshold.fill(0)
        for particle in experiment.particles.of_time_point(time_point):
            threshold[int(particle.z), int(particle.y), int(particle.x)] = 255
        morphology.distance_transform_edt(threshold.max() - threshold, sampling=sampling, distances=distances)
        distances[distances > max_distance] = max_distance
        distances_8bit = thresholding.image_to_8bit(distances)
        tifffile.imsave(path.join(output_folder, "from-points-t%03d.tif" % time_point.time_point_number()),
                        distances_8bit)


