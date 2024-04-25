"""" Loads images and positions into tensors and can write these tensors into TFR files"""
import os
from typing import Tuple, List
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

    linked = image_with_positions.linked

    return image, target_image, label, target_label, distances, linked


def tf_load_images_with_links(i: int, image_with_positions_list: List[_ImageWithLinks], time_window: List[int]
                              ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    image, target_image, label, target_label, distances, linked = tf.py_function(
        partial(load_images_with_links, image_with_positions_list=image_with_positions_list,
        time_window=time_window), [i],
        (tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.bool))

    return image, target_image, label, target_label, distances, linked
