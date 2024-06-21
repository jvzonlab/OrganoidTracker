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
from typing import List, Iterable, Tuple

from functools import partial

import keras
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

from organoid_tracker.neural_network import Tensor, image_transforms
from organoid_tracker.neural_network.dataset_transforms import ShufflingDataset, RepeatingDataset
from organoid_tracker.neural_network.link_detection_cnn.ImageWithLinks_to_tensor_loader import \
    load_images_with_links
from organoid_tracker.neural_network.link_detection_cnn.training_data_creator import _ImageWithLinks


class _TorchDataset(IterableDataset):

    _image_with_divisions_list: List[_ImageWithLinks]
    _time_window: Tuple[int, int]
    _patch_shape_zyx: Tuple[int, int, int]
    _perturb: bool

    _calculated_length: int

    def __init__(self, image_with_division_list: List[_ImageWithLinks], time_window: Tuple[int, int],
                 patch_shape_zyx: Tuple[int, int, int], perturb: bool):
        self._image_with_divisions_list = image_with_division_list
        self._time_window = time_window
        self._patch_shape_zyx = patch_shape_zyx
        self._perturb = perturb

        # Calculate the length once, to avoid recalculating it every time __len__ is called
        self._calculated_length = sum(len(image.distances) for image in image_with_division_list)

    def __iter__(self) -> Iterable[Tuple[List[Tensor], bool]]:
        # Output shape is iterable of (crops, target_crops, distances) -> linked
        for image_with_links in self._image_with_divisions_list:
            image, target_image, label, target_label, distances, linked =\
                load_images_with_links(image_with_links, self._time_window)

            image = normalize(image)
            target_image = normalize(target_image)

            for crops, target_crops, distances, linked in generate_patches_links(image, target_image,
                     label, target_label, distances, linked, self._patch_shape_zyx, self._perturb):
                if self._perturb:
                    crops, target_crops = apply_noise(crops, target_crops)
                crops, target_crops = add_3d_coord(crops, target_crops, distances)
                yield [crops, target_crops, distances], linked

    def __len__(self) -> int:
        return self._calculated_length


# Creates training and validation data from an image_with_positions_list
def training_data_creator_from_raw(images_with_links_list: List[_ImageWithLinks], time_window: Tuple[int, int],
                                   patch_shape: Tuple[int, int, int], batch_size: int, mode: str,
                                   split_proportion: float = 0.8, buffer: int = 2000, perturb=True):

    # split dataset in validation and training part
    if mode == "train":
        images_with_links_list = images_with_links_list[:round(split_proportion * len(images_with_links_list))]
    elif mode == "validation":
        images_with_links_list = images_with_links_list[round(split_proportion * len(images_with_links_list)):]

    dataset = _TorchDataset(images_with_links_list, time_window, patch_shape, perturb)

    if mode == "train":
        dataset = ShufflingDataset(dataset, buffer_size=buffer)
    dataset = RepeatingDataset(dataset)
    return DataLoader(dataset, batch_size=batch_size)


# Normalizes image data
def normalize(image):
    return keras.ops.divide(keras.ops.subtract(image, keras.ops.min(image)), keras.ops.subtract(keras.ops.max(image), keras.ops.min(image)))


def generate_patches_links(image, target_image, label, target_label, distances, linked, patch_shape, perturb):
    padding = [[patch_shape[0] // 2, patch_shape[0] // 2],
               [patch_shape[1], patch_shape[1]],
               [patch_shape[2], patch_shape[2]],
               [0, 0]]

    image_padded = keras.ops.pad(image, padding)
    target_image_padded = keras.ops.pad(target_image, padding)

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

        combined_init_crops = keras.ops.concatenate([init_crop, init_target_crop], axis=-1)

        if perturb:
            random = keras.ops.random.uniform((1,))
            combined_init_crops, distance = keras.ops.cond(random<0.99,
                                                    lambda: apply_random_flips(combined_init_crops, distance),
                                                    lambda: apply_random_perturbations_stacked(combined_init_crops, distance))
        else:
            distance = keras.ops.cast(distance, "float32")

        # second crop of the center region
        combined_crops = combined_init_crops[:,
               keras.ops.cast(patch_shape[1] / 2, "int32"): keras.ops.cast(patch_shape[1] / 2, "int32") + patch_shape[1],
               keras.ops.cast(patch_shape[2] / 2, "int32"): keras.ops.cast(patch_shape[2] / 2, "int32") + patch_shape[2], :]

        return combined_crops, distance

    distances = keras.ops.cast(distances, "int32")
    both_labels = keras.ops.stack([label, target_label, distances], axis=-1)
    for i in range(len(both_labels)):
        combined_crops, distance = single_patch(both_labels[i])
        stacked_crop = combined_crops[:, :, :, :keras.ops.shape(image)[3]]
        stacked_target_crop = combined_crops[:, :, :, keras.ops.shape(image)[3]:]
        yield stacked_crop, stacked_target_crop, distance, linked[i]


def apply_random_perturbations_stacked(stacked, distance):
    image_shape = keras.ops.cast(keras.ops.shape(stacked), "float32")

    transforms = []
    # random rotation in xy
    angle = keras.ops.random.uniform([], -np.pi, np.pi)
    transform = image_transforms.angles_to_projective_transforms(
        angle, image_shape[1], image_shape[2])
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

    stacked = keras.ops.image.affine_transform(stacked, compose_transforms, interpolation='bilinear', data_format='channels_last')

    # transform displacement vector

    distance = keras.ops.cast(distance, "float32")
    new_angle = keras.ops.arctan2(distance[2], distance[1]) + angle
    xy_length = keras.ops.sqrt(keras.ops.square(distance[1])+keras.ops.square(distance[2]))

    y_dist = xy_length*keras.ops.cos(new_angle)/scale
    x_dist = xy_length*keras.ops.sin(new_angle)/scale

    # Convert from zero-dim tensor into 1-dim tensor (otherwise keras.ops.concatenate will fail on the PyTorch backend)
    distance_0 = keras.ops.reshape(distance[0], newshape=(1,))
    y_dist = keras.ops.reshape(y_dist, newshape=(1,))
    x_dist = keras.ops.reshape(x_dist, newshape=(1,))

    distance_new = keras.ops.concatenate([distance_0, y_dist, x_dist], axis=0)
    #keras.ops.print(distance_new)

    return stacked, distance_new

def apply_random_flips(stacked, distance):
    random = keras.ops.random.uniform((1,))

    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[1]), lambda: stacked)
    distance = keras.ops.cond(random<0.5, lambda: distance * keras.ops.flip([1, -1, 1]), lambda: distance)

    random = keras.ops.random.uniform((1,))

    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[2]), lambda: stacked)
    distance = keras.ops.cond(random<0.5, lambda: distance * keras.ops.flip([1, 1, -1]), lambda: distance)

    distance = keras.ops.cast(distance, "float32")

    return stacked, distance

def random_flip_z(image, target_image, distances, linked):
    random = keras.ops.random.uniform((1,))

    image = keras.ops.cond(random<0.5, lambda: keras.ops.flip(image, axis=[0]), lambda: image)
    target_image = keras.ops.cond(random<0.5, lambda: keras.ops.flip(target_image, axis=[0]), lambda: target_image)
    distances = keras.ops.cond(random<0.5, lambda: distances * keras.ops.convert_to_tensor([-1., 1., 1.]), lambda: distances)

    return image, target_image, distances, linked


def apply_noise(image, target_image):
    # take power of image to increase or reduce contrast
    random_mul = keras.ops.random.uniform((1,), minval=0.7, maxval=1.3)
    image = keras.ops.power(image, random_mul)
    target_image = keras.ops.power(target_image, random_mul)

    # take a random decay constant (biased to 1 by taking the root)
    #decay = keras.ops.sqrt(keras.ops.random.uniform((1,), minval=0.16, maxval=1))

    # let image intensity decay differently
    #scale = decay + (1-decay) * (1 - keras.ops.range(keras.ops.shape(image)[0], dtype="float32") / keras.ops.cast(keras.ops.shape(image)[0], "float32"))
    #image = keras.ops.reshape(scale, shape=(keras.ops.shape(image)[0], 1, 1, 1)) * image
    #target_image = keras.ops.reshape(scale, shape=(keras.ops.shape(target_image)[0], 1, 1, 1)) * target_image

    return image, target_image


def add_3d_coord(image, target_image, distances):
    image = _add_3d_coord(image, distances)
    target_image = _add_3d_coord(target_image, distances, reversable=True)
    return image, target_image


def _add_3d_coord(image, offset, reversable = False):

    im_shape = keras.ops.shape(image)
    if reversable:
        z = keras.ops.abs(keras.ops.arange(-im_shape[0]//2, im_shape[0]//2, dtype='float32') + keras.ops.cast(offset[0], dtype='float32'))
        y = keras.ops.abs(keras.ops.arange(-im_shape[1]//2, im_shape[1]//2, dtype='float32') + keras.ops.cast(offset[1], dtype='float32'))
        x = keras.ops.abs(keras.ops.arange(-im_shape[2]//2, im_shape[2]//2, dtype='float32') + keras.ops.cast(offset[2], dtype='float32'))
    else:
        z = keras.ops.abs(keras.ops.arange(-im_shape[0]//2, im_shape[0]//2, dtype='float32') - keras.ops.cast(offset[0], dtype='float32'))
        y = keras.ops.abs(keras.ops.arange(-im_shape[1]//2, im_shape[1]//2, dtype='float32') - keras.ops.cast(offset[1], dtype='float32'))
        x = keras.ops.abs(keras.ops.arange(-im_shape[2]//2, im_shape[2]//2, dtype='float32') - keras.ops.cast(offset[2], dtype='float32'))

    Z, Y, X = keras.ops.meshgrid(z, y, x, indexing='ij')

    Z = keras.ops.expand_dims(Z, axis=-1)/keras.ops.cast(im_shape[0],  dtype='float32')
    Y = keras.ops.expand_dims(Y, axis=-1)/keras.ops.cast(im_shape[1],  dtype='float32')
    X = keras.ops.expand_dims(X, axis=-1)/keras.ops.cast(im_shape[2],  dtype='float32')

    if reversable:
        image = keras.ops.concatenate([X, Y, Z, image], axis=-1)
    else:
        image = keras.ops.concatenate([image, Z, Y, X], axis=-1)

    return image


# scale images
def scale(image, target_image, label, target_label, distances, linked, scale = 1):
    if scale != 1:

        transform = keras.ops.convert_to_tensor([[scale, 0., 0,
                                           0., scale, 0., 0.,
                                           0.]], dtype="float32")

        new_size = [divide_and_round(keras.ops.shape(image)[1], scale),
                    divide_and_round(keras.ops.shape(image)[2], scale)]
        image = tfa.image.transform(image, transform, interpolation='BILINEAR',
                                    output_shape=new_size)

        new_size = [divide_and_round(keras.ops.shape(target_image)[1], scale),
                    divide_and_round(keras.ops.shape(target_image)[2], scale)]
        target_image = tfa.image.transform(target_image, transform, interpolation='BILINEAR',
                                           output_shape=new_size)

        position_scaling = [1, scale, scale]
        label = divide_and_round(label, position_scaling)
        target_label = divide_and_round(target_label, position_scaling)
        distances = divide_and_round(distances, position_scaling)

    return image, target_image, label, target_label, distances, linked


def divide_and_round(tensor, scale):
    tensor = keras.ops.divide(keras.ops.cast(tensor, dtype="float32"), scale)

    return keras.ops.cast(keras.ops.round(tensor),  dtype="int32")
