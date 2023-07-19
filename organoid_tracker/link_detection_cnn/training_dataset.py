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

import numpy
import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
import numpy as np

from organoid_tracker.link_detection_cnn.ImageWithLinks_to_tensor_loader import tf_load_images_with_links
from organoid_tracker.link_detection_cnn.training_data_creator import _ImageWithLinks


# Creates training and validation data from an image_with_positions_list

def training_data_creator_from_raw(tf_load_images_with_links_list: List[_ImageWithLinks], time_window: List[int],
                                   patch_shape: List[int], batch_size: int, mode: str, split_proportion: float = 0.8, buffer: int = 2000, perturb_validation=True):

    dataset = tf.data.Dataset.range(len(tf_load_images_with_links_list))
    len_dataset = len(dataset)

    # split dataset in validation and training part
    if mode == 'train':
        dataset = dataset.take(round(split_proportion * len_dataset))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(round(split_proportion * 0.5 * len_dataset), reshuffle_each_iteration=True)

    elif mode == 'validation':
        dataset = dataset.skip(round(split_proportion * len_dataset))
        dataset = dataset.repeat()

    # Load data
    dataset = dataset.map(partial(tf_load_images_with_links, image_with_positions_list=tf_load_images_with_links_list,
                                  time_window=time_window), num_parallel_calls=12)

    # Normalize images
    #scale_down = 2
    dataset = dataset.map(normalize)
    #dataset = dataset.map(partial(scale, scale=scale_down))

    # Repeat images (as perturbations will be made)
    dataset = dataset.flat_map(partial(repeat, repeats=1))

    if mode == 'train':
        # generate multiple patches from image
        dataset = dataset.flat_map(partial(generate_patches_links, patch_shape=patch_shape, perturb=True))
        #dataset = dataset.map(random_flip_z)
        dataset = dataset.map(apply_noise)
        dataset = dataset.map(add_3d_coord)
        # create random batches
        dataset = dataset.shuffle(buffer_size=buffer)
        dataset = dataset.batch(batch_size)

    elif mode == 'validation':
        #dataset = dataset.flat_map(partial(repeat, repeats=1))
        dataset = dataset.flat_map(partial(generate_patches_links, patch_shape=patch_shape, perturb=perturb_validation))
        if perturb_validation:
            #dataset = dataset.map(random_flip_z)
            dataset = dataset.map(apply_noise)
        dataset = dataset.map(add_3d_coord)
        dataset = dataset.batch(batch_size)

    dataset = dataset.map(format)

    dataset.prefetch(1)

    return dataset

# Normalizes image data
def normalize(image, target_image, label, target_label, distances, linked):
    image = tf.divide(tf.subtract(image, tf.reduce_min(image)), tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))
    target_image = tf.divide(tf.subtract(target_image, tf.reduce_min(target_image)), tf.subtract(tf.reduce_max(target_image), tf.reduce_min(target_image)))
    return image, target_image, label, target_label, distances, linked

# Repeats
def repeat(image, target_image, label, target_label, distances, linked, repeats=5):
    dataset = tf.data.Dataset.from_tensors((image, target_image, label, target_label, distances, linked))
    dataset = dataset.repeat(repeats)

    return dataset


def format(image, target_image, distances, linked):
    return ({'input_1': image, 'input_2': target_image, 'input_distances': distances}, {'out':linked})


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


def generate_patches_links(image, target_image, label, target_label, distances, linked, patch_shape, perturb):
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
            random = tf.random.uniform((1,))
            combined_init_crops, distance = tf.cond(random<0.99,
                                                    lambda: apply_random_flips(combined_init_crops, distance),
                                                    lambda: apply_random_perturbations_stacked(combined_init_crops, distance))
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
                                   tf.data.Dataset.from_tensor_slices(distances),
                                   tf.data.Dataset.from_tensor_slices(linked)))

    return dataset


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
    #tf.print(distance_new)

    return stacked, distance_new

def apply_random_flips(stacked, distance):
    random = tf.random.uniform((1,))

    stacked = tf.cond(random<0.5, lambda: tf.reverse(stacked, axis=[1]), lambda: stacked)
    distance = tf.cond(random<0.5, lambda: distance * tf.constant([1, -1, 1]), lambda: distance)

    random = tf.random.uniform((1,))

    stacked = tf.cond(random<0.5, lambda: tf.reverse(stacked, axis=[2]), lambda: stacked)
    distance = tf.cond(random<0.5, lambda: distance * tf.constant([1, 1, -1]), lambda: distance)

    distance = tf.cast(distance, tf.float32)

    return stacked, distance

def random_flip_z(image, target_image, distances, linked):
    random = tf.random.uniform((1,))

    image = tf.cond(random<0.5, lambda: tf.reverse(image, axis=[0]), lambda: image)
    target_image = tf.cond(random<0.5, lambda: tf.reverse(target_image, axis=[0]), lambda: target_image)
    distances = tf.cond(random<0.5, lambda: distances * tf.constant([-1., 1., 1.]), lambda: distances)

    return image, target_image, distances, linked


def apply_noise(image, target_image, distances, linked):
    # take power of image to increase or reduce contrast
    random_mul = tf.random.uniform((1,), minval=0.7, maxval=1.3)
    image = tf.pow(image, random_mul)
    target_image = tf.pow(target_image, random_mul)

    # take a random decay constant (biased to 1 by taking the root)
    #decay = tf.sqrt(tf.random.uniform((1,), minval=0.16, maxval=1))

    # let image intensity decay differently
    #scale = decay + (1-decay) * (1 - tf.range(tf.shape(image)[0], dtype=tf.float32) / tf.cast(tf.shape(image)[0], tf.float32))
    #image = tf.reshape(scale, shape=(tf.shape(image)[0], 1, 1, 1)) * image
    #target_image = tf.reshape(scale, shape=(tf.shape(target_image)[0], 1, 1, 1)) * target_image

    return image, target_image, distances, linked


def add_3d_coord(image, target_image, distances, linked):
    image = _add_3d_coord(image, distances)
    target_image = _add_3d_coord(target_image, distances, reversable=True)
    return image, target_image, distances, linked


def _add_3d_coord(image, offset, reversable = False):

    im_shape = tf.shape(image)
    if reversable:
        z = tf.abs(tf.range(-im_shape[0]//2, im_shape[0]//2, dtype='float32') + tf.cast(offset[0], dtype='float32'))
        y = tf.abs(tf.range(-im_shape[1]//2, im_shape[1]//2, dtype='float32') + tf.cast(offset[1], dtype='float32'))
        x = tf.abs(tf.range(-im_shape[2]//2, im_shape[2]//2, dtype='float32') + tf.cast(offset[2], dtype='float32'))
    else:
        z = tf.abs(tf.range(-im_shape[0]//2, im_shape[0]//2, dtype='float32') - tf.cast(offset[0], dtype='float32'))
        y = tf.abs(tf.range(-im_shape[1]//2, im_shape[1]//2, dtype='float32') - tf.cast(offset[1], dtype='float32'))
        x = tf.abs(tf.range(-im_shape[2]//2, im_shape[2]//2, dtype='float32') - tf.cast(offset[2], dtype='float32'))

    Z, Y, X = tf.meshgrid(z, y, x, indexing='ij')

    Z = tf.expand_dims(Z, axis=-1)/tf.cast(im_shape[0],  dtype='float32')
    Y = tf.expand_dims(Y, axis=-1)/tf.cast(im_shape[1],  dtype='float32')
    X = tf.expand_dims(X, axis=-1)/tf.cast(im_shape[2],  dtype='float32')

    if reversable:
        image = tf.concat([X, Y, Z, image], axis=-1)
    else:
        image = tf.concat([image, Z, Y, X], axis=-1)

    return image


# scale images
def scale(image, target_image, label, target_label, distances, linked, scale = 1):
    if scale != 1:

        transform = tf.convert_to_tensor([[scale, 0., 0,
                                           0., scale, 0., 0.,
                                           0.]], dtype=tf.float32)

        new_size = [divide_and_round(tf.shape(image)[1], scale),
                    divide_and_round(tf.shape(image)[2], scale)]
        image = tfa.image.transform(image, transform, interpolation='BILINEAR',
                                    output_shape=new_size)

        new_size = [divide_and_round(tf.shape(target_image)[1], scale),
                    divide_and_round(tf.shape(target_image)[2], scale)]
        target_image = tfa.image.transform(target_image, transform, interpolation='BILINEAR',
                                           output_shape=new_size)

        position_scaling = [1, scale, scale]
        label = divide_and_round(label, position_scaling)
        target_label = divide_and_round(target_label, position_scaling)
        distances = divide_and_round(distances, position_scaling)

    return image, target_image, label, target_label, distances, linked


def divide_and_round(tensor, scale):
    tensor = tf.divide(tf.cast(tensor, dtype=tf.float32), scale)

    return tf.cast(tf.round(tensor),  dtype=tf.int32)





