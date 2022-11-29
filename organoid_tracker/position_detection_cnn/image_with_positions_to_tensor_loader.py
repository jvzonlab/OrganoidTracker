"""" Loads images and positions into tensors and can write these tensors into TFR files"""
import os
from typing import Tuple, List

import numpy
import tensorflow as tf
import numpy as np
from functools import partial
from organoid_tracker.position_detection_cnn.training_data_creator import _ImageWithPositions


_CROP_PADDING_XYZ = numpy.array([10, 10, 1])


def load_images_with_positions(i, image_with_positions_list: List[_ImageWithPositions], time_window=(0, 0), crop=True):

    image_with_positions = image_with_positions_list[i]

    # Image is zyxt, label is zyx
    image = image_with_positions.load_image_time_stack(time_window)

    if crop:
        # Find coords to crop to
        coords = image_with_positions.xyz_positions
        min_coords_xyz = np.amin(coords, axis=0) - _CROP_PADDING_XYZ
        max_coords_xyz = np.amax(coords, axis=0) + _CROP_PADDING_XYZ
        min_coords_xyz = np.clip(min_coords_xyz, [0, 0, 0], [image.shape[2], image.shape[1], image.shape[0]])
        max_coords_xyz = np.clip(max_coords_xyz, [0, 0, 0], [image.shape[2], image.shape[1], image.shape[0]])

        # Crop in x and y. We make a copy of the crop to save memory: if you make a crop in numpy, it is just a
        # reference to the original array. So the original array, which may be large, is kept in memory. To avoid
        # that, we copy the small crop, which forces numpy to allocate a new, smaller array, and allows it to delete
        # the original, large array. This was necessary to be able to train on 2048 by 2048 images with 16 GB of RAM.
        image = image[:, min_coords_xyz[1]:max_coords_xyz[1], min_coords_xyz[0]:max_coords_xyz[0]].copy()

        # Create labels (for the crop only)
        label = image_with_positions.create_labels(image_offset_zyx=(0, min_coords_xyz[1], min_coords_xyz[0]), image_size_zyx=image.shape[0:3])

        # Zero out in z (we don't crop, to preserve z-coord for CoordConv)
        if min_coords_xyz[2] > 1:
            image[0:min_coords_xyz[2] - 1].fill(0)
        if max_coords_xyz[2] < image.shape[0] - 2:
            image[max_coords_xyz[2] + 1:].fill(0)
    else:
        label = image_with_positions.create_labels(image_size_zyx=image.shape[0:3])

    return image, label


def load_images(i, image_with_positions_list: List[_ImageWithPositions], time_window=(0, 0)):
    image_with_positions = image_with_positions_list[i]

    image = image_with_positions.load_image_time_stack(time_window)

    return image


def tf_load_images(i, image_with_positions_list: List[_ImageWithPositions], time_window=[0, 0]):
    image = tf.py_function(
        partial(load_images, image_with_positions_list=image_with_positions_list,
                time_window=time_window), [i],
        tf.float32)

    return image


def tf_load_images_with_positions(i, image_with_positions_list: List[_ImageWithPositions], time_window=(0, 0)):
    image, label = tf.py_function(
        partial(load_images_with_positions, image_with_positions_list=image_with_positions_list,
                time_window=time_window), [i],
        (tf.float32, tf.float32))
    # add channel dimension to labels
    label = tf.expand_dims(label, axis=-1)

    return image, label


def dataset_writer(image_with_positions_list: List[_ImageWithPositions], time_window: Tuple[int, int], shards=10):
    dataset = tf.data.Dataset.range(len(image_with_positions_list))

    # load and serialize data
    dataset = dataset.map(partial(tf_load_images_with_positions, image_with_positions_list=image_with_positions_list,
                                  time_window=time_window), num_parallel_calls=8)
    #dataset = dataset.map(blur_labels)

    dataset_image = dataset.map(serialize_data_image)
    dataset_label = dataset.map(serialize_data_label)

    # will contain the filenames
    image_files = []
    label_files = []

    folder = "/TFR_folder"
    if not os.path.exists(folder):
        os.mkdir(folder)

    # split data into multiple TFR files, so they can be accessed more efficiently
    for i in range(shards):
        dataset_image_shard = dataset_image.shard(shards, i)
        file_name = folder + "/images{0}".format(i + 1) + '.tfrecord'
        image_files.append(file_name)
        writer = tf.data.experimental.TFRecordWriter(file_name)
        writer.write(dataset_image_shard)

        print('TFR image file {}/{}'.format(i + 1, shards))

        dataset_label_shard = dataset_label.shard(shards, i)
        file_name = folder + "/labels{0}".format(i + 1) + '.tfrecord'
        label_files.append(file_name)
        writer = tf.data.experimental.TFRecordWriter(file_name)
        writer.write(dataset_label_shard)

        print('TFR labels file {}/{}'.format(i + 1, shards))

    return image_files, label_files


def serialize_data_image(image, label):
    return tf.io.serialize_tensor(image, name='image')


def serialize_data_label(image, label):
    return tf.io.serialize_tensor(label, name='label')
