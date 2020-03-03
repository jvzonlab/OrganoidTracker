"""Starting point for the Gaussian detector: from simple cell positions to full cell shapes."""
from typing import Callable

import numpy

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Images
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.shape import GaussianShape, FAILED_SHAPE
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.util import bits
from organoid_tracker.position_detection import thresholding, watershedding, gaussian_fit


def perform_for_experiment(experiment: Experiment, *, threshold_block_size: int,
                           gaussian_fit_smooth_size: int, cluster_detection_erosion_rounds: int,
                           call_after_time_point: Callable[[TimePoint], type(None)] = lambda time_point: ...):
    for time_point in experiment.time_points():
        _perform_for_time_point(experiment.images, experiment.positions, experiment.position_data, time_point,
                                threshold_block_size, gaussian_fit_smooth_size, cluster_detection_erosion_rounds)
        call_after_time_point(time_point)


def _perform_for_time_point(images: Images, positions: PositionCollection, position_data: PositionData,
                            time_point: TimePoint, threshold_block_size: int,
                            gaussian_fit_smooth_size: int, cluster_detection_erosion_rounds: int):
    print("Working on time point " + str(time_point.time_point_number()) + "...")
    # Acquire images
    image_offset = images.offsets.of_time_point(time_point)
    positions_of_time_point = list(positions.of_time_point(time_point))
    image_positions = [None] + positions_of_time_point  # Don't use offset 0
    if not image_offset.is_zero():
        # Move all positions
        for i in range(len(image_positions)):
            if image_positions[i] is None:
                continue
            image_positions[i] = image_positions[i] - image_offset

    image_stack = images.get_image_stack(time_point)
    image_stack = bits.image_to_8bit(image_stack)

    # Create a threshold
    threshold = numpy.empty_like(image_stack, dtype=numpy.uint8)
    thresholding.advanced_threshold(image_stack, threshold, threshold_block_size)

    # Labelling, calculate distance to label
    resolution = images.resolution()
    label_image = numpy.empty_like(image_stack, dtype=numpy.uint16)
    watershedding.create_labels(image_positions, label_image)
    distance_transform_to_labels = watershedding.distance_transform_to_labels(label_image, resolution.pixel_size_zyx_um)

    # Remove places from distance transform that are outside the threshold
    distance_transform_to_labels[threshold == 0] = distance_transform_to_labels.max()

    # Perform the watershed on the threshold
    watershed = watershedding.watershed_labels(threshold, distance_transform_to_labels,
                                               label_image, len(positions_of_time_point))[0]

    # Finally use that for fitting
    gaussians = gaussian_fit.perform_gaussian_mixture_fit_from_watershed(image_stack, watershed, image_positions,
                                                                         gaussian_fit_smooth_size,
                                                                         cluster_detection_erosion_rounds)
    for position, gaussian in zip(positions_of_time_point, gaussians):
        shape = FAILED_SHAPE if gaussian is None \
            else GaussianShape(gaussian
                               .translated(image_offset.x, image_offset.y, image_offset.z)
                               .translated(-position.x, -position.y, -position.z))
        linking_markers.set_shape(position_data, position, shape)
        if gaussian is None:
            print("Could not fit gaussian for " + str(position))
