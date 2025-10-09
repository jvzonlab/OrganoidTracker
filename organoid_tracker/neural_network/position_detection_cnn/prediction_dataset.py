from functools import partial
from typing import List

import keras.ops
import numpy

from organoid_tracker.neural_network import Tensor
from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import ImageWithPositions
from organoid_tracker.neural_network.position_detection_cnn.training_dataset import pad_to_patch, normalize


def predicting_data_creator(image_with_positions_list: List[ImageWithPositions], time_window, corners,
                            patch_shape, buffer, image_shape, batch_size):

    def single_sample_generator():
        for image_with_positions in image_with_positions_list:
            # Load data
            array = image_with_positions.load_image_time_stack(time_window)
            if array is None:
                continue
            array = array.astype(numpy.float32)

            # Normalize images
            array = normalize(array)

            # Split images in smaller parts to reduce memory load
            arrays = _split(array, corners=corners, patch_shape=patch_shape, buffer=buffer, image_shape=image_shape)

            yield from arrays

    # Divide entries of single_sample_generator into batches of size batch_size
    samples = list()
    for sample in single_sample_generator():
        samples.append(keras.ops.convert_to_tensor(sample))
        if len(samples) == batch_size:
            yield keras.ops.stack(samples)
            samples.clear()


def _split(image: numpy.ndarray, corners, patch_shape, buffer, image_shape) -> List[numpy.ndarray]:
    # ensure proper image shape
    image = pad_to_patch(image, image_shape)
    image = pad_to_patch(image, patch_shape)

    # add padding
    padding = numpy.concatenate([buffer, numpy.zeros((1, 2), dtype="int32")], axis=0)
    image = numpy.pad(image, padding, mode='constant', constant_values=0)

    # The shape that has to be cropped form the images, needed?
    final_shape = [patch_shape[0] + buffer[0, 0] + buffer[0, 1],
                   patch_shape[1] + buffer[1, 0] + buffer[1, 1],
                   patch_shape[2] + buffer[2, 0] + buffer[2, 1],
                   image.shape[3]]

    images = []

    for corner in corners:
        image_crop = image[corner[0]: corner[0] + final_shape[0],
                     corner[1]: corner[1] + final_shape[1],
                     corner[2]: corner[2] + final_shape[2], :]
        images.append(image_crop)

    return images
