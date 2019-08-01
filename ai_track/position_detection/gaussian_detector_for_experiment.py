"""Starting point for the Gaussian detector: from simple cell positions to full cell shapes."""
from typing import Tuple

import numpy

from ai_track.core import TimePoint
from ai_track.core.experiment import Experiment
from ai_track.core.shape import GaussianShape, UnknownShape, UNKNOWN_SHAPE
from ai_track.imaging import bits
from ai_track.position_detection import thresholding, watershedding, gaussian_fit, smoothing


def perform_for_experiment(experiment: Experiment, threshold_block_size: int,
                            gaussian_fit_smooth_size: int, cluster_detection_erosion_rounds: int):
    for time_point in experiment.time_points():
        _perform_for_time_point(experiment, time_point, threshold_block_size,
                                gaussian_fit_smooth_size, cluster_detection_erosion_rounds)


def _perform_for_time_point(experiment: Experiment, time_point: TimePoint, threshold_block_size: int,
                            gaussian_fit_smooth_size: int, cluster_detection_erosion_rounds: int):
    print("Working on time point " + str(time_point.time_point_number()) + "...")
    # Acquire images
    image_offset = experiment.images.offsets.of_time_point(time_point)
    positions = list(experiment.positions.of_time_point(time_point))
    images = experiment.get_image_stack(time_point)
    images = bits.image_to_8bit(images)

    # Create a threshold
    threshold = numpy.empty_like(images, dtype=numpy.uint8)
    thresholding.advanced_threshold(images, threshold, threshold_block_size)

    # Labelling, calculate distance to label
    resolution = experiment.images.resolution()
    label_image = numpy.empty_like(images, dtype=numpy.uint16)
    watershedding.create_labels(positions, image_offset, label_image)
    distance_transform_to_labels = watershedding.distance_transform_to_labels(label_image, resolution.pixel_size_zyx_um)

    # Remove places from distance transform that are outside the threshold
    distance_transform_to_labels[threshold == 0] = distance_transform_to_labels.max()

    # Perform the watershed on the threshold
    watershed = watershedding.watershed_labels(threshold, distance_transform_to_labels,
                                               label_image, len(positions))[0]

    # Finally use that for fitting
    gaussians = gaussian_fit.perform_gaussian_mixture_fit_from_watershed(images, watershed, gaussian_fit_smooth_size,
                                                                         cluster_detection_erosion_rounds)
    for position, gaussian in zip(positions, gaussians):
        shape = UNKNOWN_SHAPE if gaussian is None \
            else GaussianShape(gaussian
                               .translated(image_offset.x, image_offset.y, image_offset.z)
                               .translated(-position.x, -position.y, -position.z))
        experiment.positions.add(position, shape)  # This sets the shape
        if gaussian is None:
            print("Could not fit gaussian for " + str(position))
