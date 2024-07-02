"""" Loads images and positions into tensors and can write these tensors into TFR files"""
import os
from typing import Tuple, List

import numpy
import numpy as np
from functools import partial

from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import ImageWithPositions


def load_images_with_positions(image_with_positions: ImageWithPositions, time_window=(0, 0), crop=True):

    # Image is zyxt, label is zyx
    image = image_with_positions.load_image_time_stack(time_window)
    label = image_with_positions.create_labels(image.shape[0:3])

    if crop:
        coords = image_with_positions.xyz_positions
        min_coords = np.amin(coords, axis=0)  # Becomes [x, y, z]
        max_coords = np.amax(coords, axis=0)

        # Crop in x and y
        image = image[:, min_coords[1]:max_coords[1]+1, min_coords[0]:max_coords[0]+1]
        label = label[:, min_coords[1]:max_coords[1]+1, min_coords[0]:max_coords[0]+1]

        # Zero out in z (we don't crop, to preserve z-coord for CoordConv)
        #if min_coords[2] > 1:
            #image[0:min_coords[2] - 1].fill(0)
        if max_coords[2] < image.shape[0] - 2:
            image[max_coords[2] + 1:].fill(0)

    # The dtype uint16 is not supported by PyTorch, so we convert it to int16 or int32, depending on the max value
    image = _prevent_uint16_dtype(image)
    label = _prevent_uint16_dtype(label)

    return image, label


def _prevent_uint16_dtype(array: numpy.ndarray) -> numpy.ndarray:
    """Converts uint16 arrays to int16 or int32, depending on the maximum value. This is necessary because PyTorch
    does not support uint16."""
    if array.dtype == numpy.uint16:
        if array.max() <= 2 ** 15:
            array = array.astype(numpy.int16)
        else:
            array = array.astype(numpy.int32)
    return array


def load_images(i, image_with_positions_list: List[ImageWithPositions], time_window=(0, 0)):
    image_with_positions = image_with_positions_list[i]

    image = image_with_positions.load_image_time_stack(time_window)

    return image
