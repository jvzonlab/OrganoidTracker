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
from typing import List, Tuple, Iterable

from functools import partial

import keras
from torch.utils.data import IterableDataset

from organoid_tracker.neural_network import Tensor
from organoid_tracker.neural_network.division_detection_cnn.image_with_divisions_to_tensor_loader import \
    tf_load_images_with_positions
from organoid_tracker.neural_network.link_detection_cnn.training_dataset import divide_and_round
from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import ImageWithPositions


class _TorchDataset(IterableDataset):
    _image_with_divisions_list: List[ImageWithPositions]
    _time_window: Tuple[int, int]
    _patch_shape_zyx: Tuple[int, int, int]
    _perturb: bool

    _calculated_length: int

    def __init__(self, image_with_division_list: List[ImageWithPositions], time_window: Tuple[int, int],
                 patch_shape_zyx: Tuple[int, int, int], perturb: bool):
        self._image_with_divisions_list = image_with_division_list
        self._time_window = time_window
        self._patch_shape_zyx = patch_shape_zyx
        self._perturb = perturb

        # Calculate the length once, to avoid recalculating it every time __len__ is called
        self._calculated_length = sum(len(image.xyz_positions) for image in image_with_division_list)

    def __iter__(self) -> Iterable[Tuple[Tensor, bool]]:
        for image_with_divisions in self._image_with_divisions_list:
            image = image_with_divisions.load_image_time_stack(self._time_window)
            label = image_with_divisions.xyz_positions[:, [2, 1, 0]]

            image = normalize(image)

            for crop, division in generate_patches_division(image, label, self._patch_shape_zyx):
                yield crop, division

    def __len__(self) -> int:
        return self._calculated_length


# Creates training and validation data from an image_with_positions_list
def prediction_data_creator(image_with_positions_list: List[ImageWithPositions], time_window, patch_shape):
    dataset = tf.data.Dataset.range(len(image_with_positions_list))

    # Load data
    dataset = dataset.map(partial(tf_load_images_with_positions, image_with_positions_list=image_with_positions_list,
                                  time_window=time_window), num_parallel_calls=1)

    # Normalize images
    #dataset = dataset.map(partial(scale, scale=1.33))
    dataset = dataset.map(normalize)

    dataset = dataset.flat_map(partial(generate_patches_division, patch_shape=patch_shape))
    dataset = dataset.batch(1)

    dataset.prefetch(20)

    return dataset


# Normalizes image data
def normalize(image):
    image = keras.ops.divide(keras.ops.subtract(image, keras.ops.min(image)),
                             keras.ops.subtract(keras.ops.max(image), keras.ops.min(image)))
    return image


def scale(image, label, scale=1):
    if scale != 1:
        transform = tf.convert_to_tensor([[scale, 0., 0,
                                           0., scale, 0., 0.,
                                           0.]], dtype=tf.float32)

        new_size = [divide_and_round(tf.shape(image)[1], scale),
                    divide_and_round(tf.shape(image)[2], scale)]
        image = tfa.image.transform(image, transform, interpolation='BILINEAR',
                                    output_shape=new_size)

        position_scaling = [1, scale, scale]
        label = divide_and_round(label, position_scaling)

    return image, label


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


# generates multiple patches for prediction adapted from similar taring_data version (IMPOROVE/MERGE!!!)
def generate_patches_division(image, label, patch_shape):

    padding = [[patch_shape[0]//2, patch_shape[0]//2],
               [patch_shape[1], patch_shape[1]],
               [patch_shape[2], patch_shape[2]],
               [0, 0]]

    image_padded = keras.ops.pad(image, padding)

    def single_patch(center_point):
        # first, crop to larger region
        init_crop = image_padded[center_point[0]: center_point[0] + patch_shape[0],
                          center_point[1]: center_point[1] + 2*patch_shape[1],
                          center_point[2]: center_point[2] + 2*patch_shape[2], :]

        # second, crop to the center region
        crop = init_crop[:,
               keras.ops.cast(patch_shape[1] / 2, "int32"): keras.ops.cast(patch_shape[1] / 2, "int32") + patch_shape[1],
               keras.ops.cast(patch_shape[2] / 2, "int32"): keras.ops.cast(patch_shape[2] / 2, "int32") + patch_shape[2], :]

        return crop

    # create stack of crops centered around positions
    for single_label, single_division in zip(label, dividing):
        crop = single_patch(single_label)
        yield crop, single_division
