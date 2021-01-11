from typing import List
import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
import numpy as np
from tensorflow import Tensor

from organoid_tracker.position_detection_cnn.ImageWithPositions_to_tensor_loader import tf_load_images
from organoid_tracker.position_detection_cnn.split_images import corners_split
from organoid_tracker.position_detection_cnn.training_data_creator import _ImageWithPositions
from organoid_tracker.position_detection_cnn.training_dataset import pad_to_patch


def predicting_data_creator(image_with_positions_list: List[_ImageWithPositions], time_window, corners,
                            patch_shape, buffer, image_shape):
    # load data
    dataset = tf.data.Dataset.range(len(image_with_positions_list))
    dataset = dataset.map(partial(tf_load_images, image_with_positions_list=image_with_positions_list,
                                  time_window=time_window), num_parallel_calls=8)

    # Normalize images
    dataset = dataset.map(normalize)

    # Split images in smaller parts to reduce memory load
    dataset = dataset.flat_map(partial(split, corners=corners, patch_shape=patch_shape, buffer=buffer, image_shape=image_shape))

    dataset = dataset.batch(1)
    dataset.prefetch(20)

    return dataset


def split(image, corners, patch_shape, buffer, image_shape):
    # ensure proper image shape
    image = pad_to_patch(image, image_shape)
    image = pad_to_patch(image, patch_shape)

    # add padding
    padding = tf.concat([buffer, tf.zeros((1, 2), dtype=tf.int32)], axis=0)
    image = tf.pad(image, padding, mode='CONSTANT', constant_values=0)

    # The shape that has to be cropped form the images, needed?
    final_shape = [patch_shape[0] + buffer[0, 0] + buffer[0, 1],
                   patch_shape[1] + buffer[1, 0] + buffer[1, 1],
                   patch_shape[2] + buffer[2, 0] + buffer[2, 1],
                   image._shape_as_list()[3]]
    images = []

    for corner in corners:
        image_crop = image[corner[0]: corner[0] + final_shape[0],
                     corner[1]: corner[1] + final_shape[1],
                     corner[2]: corner[2] + final_shape[2], :]
        image_crop.set_shape(final_shape)  # needed?
        images.append(image_crop)

    images = tf.stack(images, axis=0)

    return tf.data.Dataset.from_tensor_slices(images)


def normalize(image,):
    image = tf.divide(tf.subtract(image, tf.reduce_min(image)), tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))
    return (image,)
