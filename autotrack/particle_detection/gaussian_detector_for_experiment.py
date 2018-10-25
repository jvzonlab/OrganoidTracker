"""Starting point for the Gaussian detector: from simple cell positions to full cell shapes."""
from typing import Tuple

import numpy

from autotrack.core import TimePoint
from autotrack.core.experiment import Experiment
from autotrack.core.shape import GaussianShape, UnknownShape
from autotrack.particle_detection import thresholding, watershedding, gaussian_fit, smoothing


def perform_for_experiment(experiment: Experiment, **kwargs):
    for time_point in experiment.time_points():
        _perform_for_time_point(experiment, time_point, **kwargs)


def _perform_for_time_point(experiment: Experiment, time_point: TimePoint, threshold_block_size: int,
                            distance_transform_smooth_size: int, minimal_distance: Tuple[int, int, int],
                            gaussian_fit_smooth_size: int):
    print("Working on time point " + str(time_point.time_point_number()) + "...")
    # Acquire images
    particles = list(experiment.particles.of_time_point(time_point))
    images = experiment.get_image_stack(time_point)
    images = thresholding.image_to_8bit(images)

    # Create a threshold
    images_smoothed = smoothing.get_smoothed(images, int(threshold_block_size / 2))
    threshold = numpy.empty_like(images, dtype=numpy.uint8)
    minimal_distance_zyx = (minimal_distance[2], minimal_distance[1], minimal_distance[0])
    thresholding.advanced_threshold(images, images_smoothed, threshold, threshold_block_size, minimal_distance_zyx,
                                    particles)

    # Labelling, calculate distance to label
    resolution = experiment.image_loader().get_resolution()
    label_image = numpy.empty_like(images, dtype=numpy.uint16)
    watershedding.create_labels(particles, label_image)
    distance_transform_to_labels = watershedding.distance_transform_to_labels(label_image, resolution.pixel_size_zyx_um)

    # Distance transform to edge, combine with distance transform to labels
    distance_transform = numpy.empty_like(images, dtype=numpy.float64)
    watershedding.distance_transform(threshold, distance_transform, resolution.pixel_size_zyx_um)
    smoothing.smooth(distance_transform, distance_transform_smooth_size)
    distance_transform += distance_transform_to_labels

    # Perform the watershed on the rough threshold
    watershed = watershedding.watershed_labels(threshold, distance_transform.max() - distance_transform,
                                               label_image, len(particles))[0]

    # Create a basic threshold for better reconstruction of cell shape
    threshold = numpy.empty_like(images, dtype=numpy.uint8)
    thresholding.adaptive_threshold(images_smoothed, threshold, threshold_block_size)
    ones = numpy.ones_like(images, dtype=numpy.uint8)
    watershed = watershedding.watershed_labels(threshold, ones, watershed, watershed.max())[0].astype(numpy.int32)

    # Finally use that for fitting
    gaussians = gaussian_fit.perform_gaussian_mixture_fit_from_watershed(images, watershed, gaussian_fit_smooth_size)
    for particle, gaussian in zip(particles, gaussians):
        shape = UnknownShape() if gaussian is None \
            else GaussianShape(gaussian.translated(-particle.x, -particle.y, -particle.z))
        experiment.particles.add(particle, shape)
        if gaussian is None:
            print("Could not fit gaussian for " + str(particle))
