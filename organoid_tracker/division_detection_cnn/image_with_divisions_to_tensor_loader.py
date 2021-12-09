"""" Loads images and positions into tensors and can write these tensors into TFR files"""
import os
from typing import List
import tensorflow as tf
from functools import partial

from organoid_tracker.division_detection_cnn.training_data_creator import _ImageWithDivisions
from organoid_tracker.position_detection_cnn.training_data_creator import _ImageWithPositions

# Loads images with position and division information
def load_images_with_divisions(i, image_with_positions_list: List[_ImageWithDivisions], time_window=[0, 0], create_labels=False):

    image_with_positions = image_with_positions_list[i]

    image = image_with_positions.load_image_time_stack(time_window)
    # change positions in to label image (pixels in image)
    if create_labels:
        label = image_with_positions.create_labels(image.shape[0:3])
    # reorder the position information to zyx
    else:
        label = image_with_positions.xyz_positions
        label = label[:, [2,1,0]]
    # extract division info
    dividing = image_with_positions._dividing

    return image, label, dividing

# tensorflow wrapper to laod image + division data
def tf_load_images_with_divisions(i, image_with_positions_list: List[_ImageWithPositions], time_window=[0, 0], create_labels=False):
    if create_labels:
        image, label, dividing = tf.py_function(
            partial(load_images_with_divisions, image_with_positions_list=image_with_positions_list,
                time_window=time_window, create_labels=create_labels), [i],
            (tf.float32, tf.float32, tf.bool))
    else:
        image, label, dividing = tf.py_function(
            partial(load_images_with_divisions, image_with_positions_list=image_with_positions_list,
                time_window=time_window, create_labels=create_labels), [i],
            (tf.float32, tf.int32, tf.bool))

    return image, label, dividing


def load_images_with_positions(i, image_with_positions_list: List[_ImageWithDivisions], time_window=[0, 0], create_labels=False):

    image_with_positions = image_with_positions_list[i]

    image = image_with_positions.load_image_time_stack(time_window)
    # change positions in to label image (pixels in image)
    if create_labels:
        label = image_with_positions.create_labels(image.shape[0:3])
    # reorder the position information to zyx
    else:

        label = image_with_positions.xyz_positions
        label = label[:, [2,1,0]]

    return image, label

# tensorflow wrapper to laod image + positions
def tf_load_images_with_positions(i, image_with_positions_list: List[_ImageWithPositions], time_window=[0, 0], create_labels=False):
    if create_labels:
        image, label = tf.py_function(
            partial(load_images_with_positions, image_with_positions_list=image_with_positions_list,
                time_window=time_window, create_labels=create_labels), [i],
            (tf.float32, tf.float32))
    else:
        image, label = tf.py_function(
            partial(load_images_with_positions, image_with_positions_list=image_with_positions_list,
                time_window=time_window, create_labels=create_labels), [i],
            (tf.float32, tf.int32))

    return image, label



def dataset_writer(image_with_positions_list: List[_ImageWithPositions], time_window, shards=10):
    dataset = tf.data.Dataset.range(len(image_with_positions_list))

    # load and serialize data
    dataset = dataset.map(partial(tf_load_images_with_divisions, image_with_positions_list=image_with_positions_list,
                                  time_window=time_window), num_parallel_calls=8)
    #dataset = dataset.map(blur_labels)

    dataset_image = dataset.map(serialize_data_image)
    dataset_label = dataset.map(serialize_data_label)
    dataset_dividing = dataset.map(serialize_data_dividing)

    # will contain the filenames
    image_files = []
    label_files = []
    dividing_files = []

    folder = "TFR_folder"
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

        dataset_dividing_shard = dataset_dividing.shard(shards, i)
        file_name = folder + "/dividing{0}".format(i + 1) + '.tfrecord'
        dividing_files.append(file_name)
        writer = tf.data.experimental.TFRecordWriter(file_name)
        writer.write(dataset_dividing_shard)

        print('TFR dividing file {}/{}'.format(i + 1, shards))

    return image_files, label_files, dividing_files


def serialize_data_image(image, label, dividing):
    return tf.io.serialize_tensor(image, name='image')


def serialize_data_label(image, label, dividing):
    return tf.io.serialize_tensor(label, name='label')

def serialize_data_dividing(image, label, dividing):
    return tf.io.serialize_tensor(dividing, name='dividing')