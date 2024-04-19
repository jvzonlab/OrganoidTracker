"""" Loads images and positions into tensors and can write these tensors into TFR files"""
import os
from typing import Tuple, List
import tensorflow as tf
from functools import partial


from organoid_tracker.neural_network.link_detection_cnn.training_data_creator import _ImageWithLinks
from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import ImageWithPositions


def load_images_with_links(i, image_with_positions_list: List[_ImageWithLinks], time_window=[0, 0]):
    image_with_positions = image_with_positions_list[i]

    image = image_with_positions.load_image_time_stack(time_window)

    time_window_target = []
    time_window_target.append(-time_window[1])
    time_window_target.append(-time_window[0])
    target_image = image_with_positions.load_image_time_stack(time_window_target, delay=1)

    label = image_with_positions.xyz_positions
    label = label[:, [2,1,0]]

    target_label = image_with_positions.target_xyz_positions
    target_label = target_label[:, [2,1,0]]

    distances = image_with_positions.distances
    distances = distances[:, [2,1,0]]

    linked = image_with_positions._linked

    return image, target_image, label, target_label, distances, linked


def tf_load_images_with_links(i: int, image_with_positions_list: List[_ImageWithLinks], time_window: List[int]
                              ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    image, target_image, label, target_label, distances, linked = tf.py_function(
        partial(load_images_with_links, image_with_positions_list=image_with_positions_list,
        time_window=time_window), [i],
        (tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.bool))

    return image, target_image, label, target_label, distances, linked


def dataset_writer(image_with_positions_list: List[ImageWithPositions], time_window: List[int], shards: int = 10):
    dataset = tf.data.Dataset.range(len(image_with_positions_list))

    # load and serialize data
    dataset = dataset.map(partial(tf_load_images_with_links, image_with_positions_list=image_with_positions_list,
                                  time_window=time_window), num_parallel_calls=8)
    #dataset = dataset.map(blur_labels)

    dataset_image = dataset.map(serialize_data_image)
    dataset_label = dataset.map(serialize_data_label)
    dataset_target_label = dataset.map(serialize_data_target_label)
    dataset_linked = dataset.map(serialize_data_linked)

    # will contain the filenames
    image_files = []
    label_files = []
    target_label_files = []
    linked_files = []

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

        dataset_target_label_shard = dataset_target_label.shard(shards, i)
        file_name = folder + "/target_labels{0}".format(i + 1) + '.tfrecord'
        target_label_files.append(file_name)
        writer = tf.data.experimental.TFRecordWriter(file_name)
        writer.write(dataset_target_label_shard)

        print('TFR target labels file {}/{}'.format(i + 1, shards))


        dataset_linked_shard = dataset_linked.shard(shards, i)
        file_name = folder + "/linked{0}".format(i + 1) + '.tfrecord'
        linked_files.append(file_name)
        writer = tf.data.experimental.TFRecordWriter(file_name)
        writer.write(dataset_linked_shard)

        print('TFR linked file {}/{}'.format(i + 1, shards))

    return image_files, label_files, linked_files


def serialize_data_image(image, label, target_label, linked):
    return tf.io.serialize_tensor(image, name='image')


def serialize_data_label(image, label, target_label,linked):
    return tf.io.serialize_tensor(label, name='label')

def serialize_data_target_label(image, label, target_label,linked):
    return tf.io.serialize_tensor(target_label, name='label')

def serialize_data_linked(image, label, target_label, linked):
    return tf.io.serialize_tensor(linked, name='linked')
