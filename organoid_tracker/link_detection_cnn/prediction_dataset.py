# File originally written by Laetitia Hebert in 2018 (Okinawa Institute of Technology, Japan). Modified by Rutger.
#
# The MIT License
#
# Copyright (c) 2018 Okinawa Institute of Science & Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Tuple, List

import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
import numpy as np

from organoid_tracker.link_detection_cnn.ImageWithLinks_to_tensor_loader import tf_load_images_with_links
from organoid_tracker.link_detection_cnn.training_data_creator import _ImageWithLinks


# Creates training and validation data from an image_with_positions_list
from organoid_tracker.link_detection_cnn.training_dataset import normalize, scale, _add_3d_coord


def prediction_data_creator(tf_load_images_with_links_list: List[_ImageWithLinks], time_window: List[int], patch_shape_zyx: List[int]):
    dataset = tf.data.Dataset.range(len(tf_load_images_with_links_list))

    # Load data
    dataset = dataset.map(partial(tf_load_images_with_links, image_with_positions_list=tf_load_images_with_links_list,
                                  time_window=time_window, crop=False), num_parallel_calls=12)


    # Normalize images
    dataset = dataset.map(normalize)

    dataset = dataset.map(drop_linked)
    dataset = dataset.flat_map(partial(generate_patches_links, patch_shape=patch_shape_zyx, perturb=False))
    dataset = dataset.map(add_3d_coord)
    dataset = dataset.map(format)

    dataset = dataset.batch(50)
    dataset.prefetch(2)

    return dataset

def drop_linked(image, target_image, label, target_label, distances, linked):
    return image, target_image, label, target_label, distances

# Normalizes image data
#def normalize(image, target_image, label, target_label, distances):
    #image = tf.divide(tf.subtract(image, tf.reduce_min(image)), tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))
    #target_image = tf.divide(tf.subtract(target_image, tf.reduce_min(target_image)), tf.subtract(tf.reduce_max(target_image), tf.reduce_min(target_image)))
    #return image, target_image, label, target_label, distances


def format(image, target_image, distances):
    return {'input_1': image, 'input_2': target_image, 'input_distances': distances}


def pad_to_patch(stacked, patch_shape):
    stacked_shape = tf.shape(stacked)

    pad_z = tf.cond(tf.less(stacked_shape[0], patch_shape[0]), lambda: patch_shape[0] - stacked_shape[0],
                    lambda: 0)
    pad_y = tf.cond(tf.less(stacked_shape[1], patch_shape[1]), lambda: patch_shape[1] - stacked_shape[1],
                    lambda: 0)
    pad_x = tf.cond(tf.less(stacked_shape[2], patch_shape[2]), lambda: patch_shape[2] - stacked_shape[2],
                    lambda: 0)
    padding = [[pad_z, 0], [pad_y, 0], [pad_x, 0], [0, 0]]

    return tf.pad(stacked, padding)


def generate_patches_links(image, target_image, label, target_label, distances, patch_shape, perturb):
    padding = [[patch_shape[0] // 2, patch_shape[0] // 2],
               [patch_shape[1], patch_shape[1]],
               [patch_shape[2], patch_shape[2]],
               [0, 0]]

    image_padded = tf.pad(image, padding)
    target_image_padded = tf.pad(target_image, padding)

    def single_patch(center_points_distance):
        center_point = center_points_distance[:, 0]
        target_center_point = center_points_distance[:, 1]
        distance = center_points_distance[:, 2]

        # first crop

        init_crop = image_padded[center_point[0]: center_point[0] + patch_shape[0],
                    center_point[1]: center_point[1] + 2 * patch_shape[1],
                    center_point[2]: center_point[2] + 2 * patch_shape[2], :]

        init_target_crop = target_image_padded[target_center_point[0]: target_center_point[0] + patch_shape[0],
                    target_center_point[1]: target_center_point[1] + 2 * patch_shape[1],
                    target_center_point[2]: target_center_point[2] + 2 * patch_shape[2], :]

        combined_init_crops = tf.concat([init_crop, init_target_crop], axis=-1)

        if perturb:
            combined_init_crops, distance = apply_random_perturbations_stacked(combined_init_crops, distance)
        else:
            distance = tf.cast(distance, tf.float32)

        # second crop of the center region
        combined_crops = combined_init_crops[:,
               tf.cast(patch_shape[1] / 2, tf.int32): tf.cast(patch_shape[1] / 2, tf.int32) + patch_shape[1],
               tf.cast(patch_shape[2] / 2, tf.int32): tf.cast(patch_shape[2] / 2, tf.int32) + patch_shape[2], :]

        return combined_crops, distance

    distances = tf.cast(distances, tf.int32)
    both_labels = tf.stack([label, target_label, distances], axis=-1)
    stacked_combined_crops, distances = tf.map_fn(single_patch, both_labels, parallel_iterations=10, fn_output_signature=(tf.float32, tf.float32))

    #tf.print(distances)

    stacked_crops = stacked_combined_crops[:, :, :, :, :tf.shape(image)[3]]
    stacked_target_crops = stacked_combined_crops[:, :, :, :, tf.shape(image)[3]:]

    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(stacked_crops),
                                   tf.data.Dataset.from_tensor_slices(stacked_target_crops),
                                   tf.data.Dataset.from_tensor_slices(distances)))

    return dataset


def add_3d_coord(image, target_image, distances):
    image = _add_3d_coord(image, distances)

    return image, _add_3d_coord(target_image, distances, reversable=True), distances


def apply_random_perturbations_stacked(stacked, distance):
    image_shape = tf.cast(tf.shape(stacked), tf.float32)

    transforms = []
    # random rotation in xy
    angle = tf.random.uniform([], -np.pi, np.pi)
    transform = tfa.image.transform_ops.angles_to_projective_transforms(
        angle, image_shape[1], image_shape[2])
    transforms.append(transform)
    # random scale 80% to 120% size
    scale = tf.random.uniform([], 0.8, 1.2, dtype=tf.float32)
    transform = tf.convert_to_tensor([[scale, 0., image_shape[1] / 2 * (1 - scale),
                                       0., scale, image_shape[2] / 2 * (1 - scale), 0.,
                                       0.]], dtype=tf.float32)
    transforms.append(transform)

    # compose rotation-scale transform
    compose_transforms = tfa.image.transform_ops.compose_transforms(transforms)
    stacked = tfa.image.transform(stacked, compose_transforms, interpolation='BILINEAR')

    # transform displacement vector
    distance = tf.cast(distance, tf.float32)
    new_angle = tf.math.atan2(distance[2], distance[1]) + angle
    xy_length = tf.sqrt(tf.square(distance[1])+tf.square(distance[2]))

    y_dist = xy_length*tf.math.cos(new_angle)/scale
    x_dist = xy_length*tf.math.sin(new_angle)/scale

    distance_new = tf.concat([distance[0], y_dist, x_dist], axis=0)

    return stacked, distance_new