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
from typing import List

import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
import numpy as np

from organoid_tracker.neural_network.division_detection_cnn.image_with_divisions_to_tensor_loader import tf_load_images_with_divisions
from organoid_tracker.neural_network.division_detection_cnn.training_data_creator import _ImageWithDivisions


# Creates training and validation data from an image_with_positions_list
def training_data_creator_from_raw(image_with_divisions_list: List[_ImageWithDivisions], time_window, patch_shape,
                                   batch_size: int, mode, split_proportion: float = 0.8, perturb=True):
    dataset = tf.data.Dataset.range(len(image_with_divisions_list))
    len_dataset = len(dataset)

    # split dataset in validation and training part
    if mode == 'train':
        dataset = dataset.take(round(split_proportion * len(dataset)))
        #dataset = dataset.shuffle(round(split_proportion * len(dataset))) #, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(round(0.1*len_dataset))
    elif mode == 'validation':
        dataset = dataset.skip(round(split_proportion * len(dataset)))
        #dataset = dataset.repeat()

    # Load data
    dataset = dataset.map(partial(tf_load_images_with_divisions, image_with_positions_list=image_with_divisions_list,
                                  time_window=time_window, create_labels=False), num_parallel_calls=12)

    # Normalize images
    dataset = dataset.map(normalize)

    # Repeat images (as perturbations will be made)
    dataset = dataset.flat_map(partial(repeat, repeats=1))

    if mode == 'train':
        # generate multiple patches from image
        dataset = dataset.flat_map(partial(generate_patches_division, patch_shape=patch_shape, perturb=perturb))
        if perturb:
            dataset = dataset.map(apply_noise)
        # create random batches
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)

    elif mode == 'validation':
        dataset = dataset.flat_map(partial(generate_patches_division, patch_shape=patch_shape, perturb=perturb))
        if perturb:
            dataset = dataset.map(apply_noise)
        dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.batch(batch_size)

    dataset.prefetch(1)

    return dataset


# Creates training and validation data from TFR files
def training_data_creator_from_TFR(images_file, labels_file, dividing_file, patch_shape: List[int],
                                   batch_size=1, mode=None, split_proportion: float = 0.8, n_images: int = 0):

    dataset_images = tf.data.TFRecordDataset(images_file, num_parallel_reads=10)
    dataset_images = dataset_images.map(lambda x: tf.io.parse_tensor(x, tf.float32))

    dataset_labels = tf.data.TFRecordDataset(labels_file, num_parallel_reads=10)
    dataset_labels = dataset_labels.map(lambda x: tf.io.parse_tensor(x, tf.int32))

    dataset_dividing = tf.data.TFRecordDataset(dividing_file, num_parallel_reads=10)
    dataset_dividing = dataset_dividing.map(lambda x: tf.io.parse_tensor(x, tf.bool))

    dataset = tf.data.Dataset.zip((dataset_images, dataset_labels, dataset_dividing))

    if mode == 'train':
        dataset = dataset.take(round(split_proportion * n_images))
        # dataset = dataset.shuffle(buffer_size=2)  # small shuffling so that each iteration is a little different
        dataset = dataset.repeat()
    elif mode == 'validation':
        dataset = dataset.skip(round(split_proportion * n_images))
        dataset = dataset.repeat()  # generate 5 patches from every image

    # Normalize images
    dataset = dataset.map(normalize)

    if mode == 'train':
        # generate multiple patches from image
        dataset = dataset.flat_map(partial(generate_patches_division, patch_shape=patch_shape, perturb=True))
        dataset = dataset.map(apply_noise)
        # create random batches
        dataset = dataset.shuffle(buffer_size=200000)
        dataset = dataset.batch(batch_size)


    elif mode == 'validation':
        #dataset = dataset.map(partial(generate_patch, patch_shape=patch_shape, batch=False))
        dataset = dataset.flat_map(partial(generate_patches_division, patch_shape=patch_shape, perturb=False))
        dataset = dataset.shuffle(buffer_size=200000)
        dataset = dataset.batch(batch_size)

    #dataset = dataset.prefetch(5)

    return dataset

# Normalizes image data
def normalize(image, label, dividing):
    image = tf.divide(tf.subtract(image, tf.reduce_min(image)), tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))
    return image, label, dividing

# pads image if smaller than the patch size
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


# Repeats
def repeat(image, label, dividing, repeats=5):
    dataset = tf.data.Dataset.from_tensors((image, label, dividing))
    dataset = dataset.repeat(repeats)

    return dataset

# generates multiple perturbed patches
def generate_patches_division(image, label, dividing, patch_shape, perturb):

    # add padding around image edge
    padding = [[patch_shape[0]//2, patch_shape[0]//2],
               [patch_shape[1], patch_shape[1]],
               [patch_shape[2], patch_shape[2]],
               [0, 0]]

    image_padded = tf.pad(image, padding)

    # Extracts single patch
    def single_patch(center_point):
        # first crop
        init_crop = image_padded[center_point[0]: center_point[0] + patch_shape[0],
                          center_point[1]: center_point[1] + 2*patch_shape[1],
                          center_point[2]: center_point[2] + 2*patch_shape[2], :]

        # apply perturbations
        if perturb:
            #init_crop = apply_random_perturbations_stacked(init_crop)
            random = tf.random.uniform((1,))
            init_crop = tf.cond(random<0.5,
                                lambda: apply_random_flips(init_crop),
                                lambda: apply_random_perturbations_stacked(init_crop))

            #init_crop = black_out(init_crop)

        # second crop of the center region
        crop = init_crop[:,
               tf.cast(patch_shape[1] / 2, tf.int32): tf.cast(patch_shape[1] / 2, tf.int32) + patch_shape[1],
               tf.cast(patch_shape[2] / 2, tf.int32): tf.cast(patch_shape[2] / 2, tf.int32) + patch_shape[2], :]

        return(crop)

    # create stack of crops centered around positions
    stacked_crops = tf.map_fn(single_patch, label, parallel_iterations=10, fn_output_signature=tf.float32)

    # creates tf.dataset from stack of crops and associated division labels
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(stacked_crops),
                        tf.data.Dataset.from_tensor_slices(dividing)))

    return dataset


def apply_random_perturbations_stacked(stacked):
    image_shape = tf.cast(tf.shape(stacked), tf.float32)

    transforms = []
    # random rotation in xy
    transform = tfa.image.transform_ops.angles_to_projective_transforms(
        tf.random.uniform([], -np.pi, np.pi), image_shape[1], image_shape[2])
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

    return stacked


def apply_random_flips(stacked):
    random = tf.random.uniform((1,))
    stacked = tf.cond(random<0.5, lambda: tf.reverse(stacked, axis=[1]), lambda: stacked)

    random = tf.random.uniform((1,))
    stacked = tf.cond(random<0.5, lambda: tf.reverse(stacked, axis=[2]), lambda: stacked)

    random = tf.random.uniform((1,))
    stacked = tf.cond(random<0.5, lambda: tf.reverse(stacked, axis=[0]), lambda: stacked)

    return stacked


def apply_noise(image, dividing):
    # add noise
    # take power of image to increase or reduce contrast
    image = tf.pow(image, tf.random.uniform((1,), minval=0.3, maxval=1.7))

    return image, dividing


