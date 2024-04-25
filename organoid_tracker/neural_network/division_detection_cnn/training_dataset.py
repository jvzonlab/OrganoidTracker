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

from functools import partial

import keras
import numpy as np

from organoid_tracker.neural_network import image_transforms
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


# Normalizes image data
def normalize(image, label, dividing):
    image = keras.ops.divide(keras.ops.subtract(image, keras.ops.min(image)), keras.ops.subtract(keras.ops.max(image), keras.ops.min(image)))
    return image, label, dividing

# pads image if smaller than the patch size
def pad_to_patch(stacked, patch_shape):
    stacked_shape = keras.ops.shape(stacked)

    pad_z = keras.ops.cond(keras.ops.less(stacked_shape[0], patch_shape[0]), lambda: patch_shape[0] - stacked_shape[0],
                    lambda: 0)
    pad_y = keras.ops.cond(keras.ops.less(stacked_shape[1], patch_shape[1]), lambda: patch_shape[1] - stacked_shape[1],
                    lambda: 0)
    pad_x = keras.ops.cond(keras.ops.less(stacked_shape[2], patch_shape[2]), lambda: patch_shape[2] - stacked_shape[2],
                    lambda: 0)
    padding = [[pad_z, 0], [pad_y, 0], [pad_x, 0], [0, 0]]

    return keras.ops.pad(stacked, padding)


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

    image_padded = keras.ops.pad(image, padding)

    # Extracts single patch
    def single_patch(center_point):
        # first crop
        init_crop = image_padded[center_point[0]: center_point[0] + patch_shape[0],
                          center_point[1]: center_point[1] + 2*patch_shape[1],
                          center_point[2]: center_point[2] + 2*patch_shape[2], :]

        # apply perturbations
        if perturb:
            #init_crop = apply_random_perturbations_stacked(init_crop)
            random = keras.ops.random.uniform((1,))
            init_crop = keras.ops.cond(random<0.5,
                                lambda: apply_random_flips(init_crop),
                                lambda: apply_random_perturbations_stacked(init_crop))

            #init_crop = black_out(init_crop)

        # second crop of the center region
        crop = init_crop[:,
               keras.ops.cast(patch_shape[1] / 2, "int32"): keras.ops.cast(patch_shape[1] / 2, "int32") + patch_shape[1],
               keras.ops.cast(patch_shape[2] / 2, "int32"): keras.ops.cast(patch_shape[2] / 2, "int32") + patch_shape[2], :]

        return(crop)

    # create stack of crops centered around positions
    stacked_crops = tf.map_fn(single_patch, label, parallel_iterations=10, fn_output_signature="float32")

    # creates tf.dataset from stack of crops and associated division labels
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(stacked_crops),
                        tf.data.Dataset.from_tensor_slices(dividing)))

    return dataset


def apply_random_perturbations_stacked(stacked):
    image_shape = keras.ops.cast(keras.ops.shape(stacked), "float32")

    transforms = []
    # random rotation in xy
    transform = image_transforms.angles_to_projective_transforms(
        keras.ops.random.uniform([], -np.pi, np.pi), image_shape[1], image_shape[2])
    transforms.append(transform)
    # random scale 80% to 120% size
    scale = keras.ops.random.uniform([], 0.8, 1.2, dtype="float32")
    transform = keras.ops.convert_to_tensor([[scale, 0., image_shape[1] / 2 * (1 - scale),
                                       0., scale, image_shape[2] / 2 * (1 - scale), 0.,
                                       0.]], dtype="float32")
    transforms.append(transform)

    # compose rotation-scale transform
    compose_transforms = image_transforms.compose_transforms(transforms)
    # convert 1 x 8 array to Z x 8 array, as every layer should get the same transformation
    compose_transforms = keras.ops.tile(compose_transforms, [keras.ops.shape(stacked)[0], 1])
    stacked = keras.ops.image.affine_transform(stacked, compose_transforms, interpolation='bilinear')

    return stacked


def apply_random_flips(stacked):
    random = keras.ops.random.uniform((1,))
    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[1]), lambda: stacked)

    random = keras.ops.random.uniform((1,))
    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[2]), lambda: stacked)

    random = keras.ops.random.uniform((1,))
    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[0]), lambda: stacked)

    return stacked


def apply_noise(image, dividing):
    # add noise
    # take power of image to increase or reduce contrast
    image = keras.ops.power(image, keras.ops.random.uniform((1,), minval=0.3, maxval=1.7))

    return image, dividing


