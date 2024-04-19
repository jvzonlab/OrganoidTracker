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
import keras.random
import numpy as np

from organoid_tracker.neural_network.position_detection_cnn.image_with_positions_to_tensor_loader import tf_load_images_with_positions
from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import ImageWithPositions

# Creates training and validation data from an image_with_positions_list
def training_data_creator_from_raw(image_with_positions_list: List[ImageWithPositions], time_window, patch_shape,
                                   batch_size: int, mode, split_proportion: float = 0.8, crop=False):

    dataset = tf.data.Dataset.range(len(image_with_positions_list))

    # split dataset in validation and training part
    if mode == 'train':
        dataset = dataset.take(round(split_proportion * len(dataset)))
        dataset = dataset.shuffle(len(dataset))
        dataset = dataset.repeat()
    elif mode == 'validation':
        dataset = dataset.skip(round(split_proportion * len(dataset)))
        dataset = dataset.repeat()

    # Load data
    dataset = dataset.map(partial(tf_load_images_with_positions, image_with_positions_list=image_with_positions_list,
                                  time_window=time_window, crop=crop), num_parallel_calls=12)

    # Normalize images
    dataset = dataset.map(normalize)

    if mode == 'train':
        # generate multiple patches from image
        dataset = dataset.flat_map(partial(generate_patches, patch_shape=patch_shape, multiplier=batch_size))
        dataset = dataset.map(apply_noise)

        # create random batches
        dataset = dataset.shuffle(buffer_size=10*batch_size)
        dataset = dataset.batch(batch_size)

    elif mode == 'validation':
        dataset = dataset.flat_map(partial(generate_patches, patch_shape=patch_shape, multiplier=batch_size, perturb=False))
        dataset = dataset.batch(batch_size)

    dataset.prefetch(2)

    return dataset



# Normalizes image data
def normalize(image, label):

    image = keras.ops.divide(keras.ops.subtract(image, keras.ops.min(image)), keras.ops.subtract(keras.ops.max(image), keras.ops.min(image)))

    return image, label


def z_shift(stacked, max_shift = 1):

    shift = keras.random.uniform(shape=[], minval= -max_shift,  maxval=max_shift, dtype="int32")

    padding = [[max_shift, max_shift+1], [0, 0], [0, 0], [0, 0]]

    stacked = keras.ops.pad(stacked, padding, constant_values=0, mode='constant')

    stacked = stacked[(max_shift+shift): -(1+max_shift-shift), :, :, :]

    return stacked


def pad_to_patch(stacked, patch_shape):
    stacked_shape = keras.ops.shape(stacked)

    pad_z = keras.ops.cond(keras.ops.less(stacked_shape[0], patch_shape[0]), lambda: patch_shape[0] - stacked_shape[0],
                    lambda: 0)
    pad_y = keras.ops.cond(keras.ops.less(stacked_shape[1], patch_shape[1]), lambda: patch_shape[1] - stacked_shape[1],
                    lambda: 0)
    pad_x = keras.ops.cond(keras.ops.less(stacked_shape[2], patch_shape[2]), lambda: patch_shape[2] - stacked_shape[2],
                    lambda: 0)

    padding = [[pad_z, 0], [0, pad_y], [0, pad_x], [0, 0]]

    return keras.ops.pad(stacked, padding, mode='constant', constant_values=0)


# generates single patch without pertubations for validation set
def generate_patch(image, label, patch_shape, batch=False, perturb=True):
    # concat in channel dimension
    stacked = keras.ops.concatenate([image, label], axis=-1)

    stacked = pad_to_patch(stacked, patch_shape)

    patch_shape = patch_shape + [keras.ops.shape(stacked)[-1]]

    # needed?
    if batch:
        patch_shape = [keras.ops.shape(stacked)[0]] + patch_shape

    stacked = tf.image.random_crop(stacked, size=patch_shape)

    image = stacked[:, :, :, :keras.ops.shape(image)[3]]
    label = stacked[:, :, :, keras.ops.shape(image)[3]:]

    return image, label

# generates multiple perturbed patches
def generate_patches(image, label, patch_shape, multiplier=20, perturb=True):
    # concat image and labels in channel dimension
    stacked = keras.ops.concatenate([image, label], axis=-1)

    # initial crop is twice the final crop size in x and y
    patch_shape_init = list(patch_shape)
    patch_shape_init[1] = 2 * patch_shape[1]
    patch_shape_init[2] = 2 * patch_shape[2]

    # if the image is smaller that the patch region then pad
    stacked = pad_to_patch(stacked, patch_shape_init)

    # add buffer region
    #padding = [[patch_shape[0]//8, patch_shape[0]//8], [patch_shape[1]//2, patch_shape[1]//2], [patch_shape[2]//2, patch_shape[2]//2], [0, 0]]
    padding = [[0, 0], [patch_shape[1] // 2, patch_shape[1] // 2],
               [patch_shape[2] // 2, patch_shape[2] // 2], [0, 0]]
    #padding = [[0, 0], [patch_shape[1], patch_shape[1]], [patch_shape[2], patch_shape[2]], [0, 0]]
    stacked = keras.ops.pad(stacked, padding, mode='constant', constant_values=0)

    # add channel dimensions
    patch_shape_init = patch_shape_init + [keras.ops.shape(stacked)[-1]]

    stacked_crops = []

    for i in range(multiplier):
        # first crop
        stacked_crop = tf.image.random_crop(stacked, size=patch_shape_init)

        # apply perturbations
        if perturb:
            #stacked_crop = apply_random_perturbations_stacked(stacked_crop)
            random = keras.random.uniform((1,))
            stacked_crop = keras.ops.cond(random<0.5,
                                          lambda: apply_random_flips(stacked_crop),
                                          lambda: apply_random_perturbations_stacked(stacked_crop))

            #stacked_crop = z_shift(stacked_crop)

        # second crop of the center region
        stacked_crop = stacked_crop[:, keras.ops.cast(patch_shape[1]/2, "int32"): keras.ops.cast(patch_shape[1]/2, "int32") + patch_shape[1],
                                    keras.ops.cast(patch_shape[2]/2, "int32"): keras.ops.cast(patch_shape[2]/2, "int32") + patch_shape[2], :]

        stacked_crops.append(stacked_crop)

    stacked_crops = keras.ops.stack(stacked_crops, axis=0)

    # split labels and images
    image_crops = stacked_crops[:, :, :, :, :keras.ops.shape(image)[3]]
    label_crops = stacked_crops[:, :, :, :, keras.ops.shape(image)[3]:]

    dataset = tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(image_crops), tf.data.Dataset.from_tensor_slices(label_crops)))

    return dataset


def apply_random_perturbations_stacked(stacked):
    image_shape = keras.ops.cast(keras.ops.shape(stacked), "float32")
    transforms = []
    # random rotation in xy
    transform = image_ops.angles_to_projective_transforms(
        keras.random.uniform([], -np.pi, np.pi), image_shape[1], image_shape[2])
    transforms.append(transform)
    # random scale 80% to 120% size
    scale = keras.random.uniform([], 0.8, 1.2, dtype="float32")
    transform = keras.ops.convert_to_tensor([[scale, 0., image_shape[1] / 2 * (1 - scale),
                                            0., scale, image_shape[2] / 2 * (1 - scale), 0.,
                                            0.]], dtype="float32")
    transforms.append(transform)

    # compose rotation-scale transform
    compose_transforms = image_ops.compose_transforms(transforms)
    stacked = keras.ops.image.affine_transform(stacked, compose_transforms, interpolation='bilinear')

    return stacked


def apply_random_flips(stacked):
    random = keras.random.uniform((1,))
    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[1]), lambda: stacked)

    random = keras.random.uniform((1,))
    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[2]), lambda: stacked)

    return stacked


def apply_noise(image, label):
    # take power of image to increase or reduce contrast
    image = keras.ops.power(image, keras.random.uniform((1,), minval=0.8, maxval=1.2))

    # take a random decay constant (biased to 1 by taking the root)
    decay = keras.ops.sqrt(keras.random.uniform((1,), minval=0.04, maxval=1))

    # let image intensity decay differently
    scale = decay + (1-decay) * (1 - keras.ops.arange(keras.ops.shape(image)[0], dtype="float32") / keras.ops.cast(keras.ops.shape(image)[0], "float32"))
    image = keras.ops.reshape(scale, shape=(keras.ops.shape(image)[0], 1, 1, 1)) * image

    return image, label

