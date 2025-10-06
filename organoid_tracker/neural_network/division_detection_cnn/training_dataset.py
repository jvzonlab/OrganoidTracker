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

import keras
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

from organoid_tracker.neural_network import image_transforms, Tensor
from organoid_tracker.neural_network.dataset_transforms import ShufflingDataset, RepeatingDataset, PrefetchingDataset
from organoid_tracker.neural_network.division_detection_cnn.training_data_creator import _ImageWithDivisions


class _TorchDataset(IterableDataset):

    _image_with_divisions_list: List[_ImageWithDivisions]
    _time_window: Tuple[int, int]
    _patch_shape_zyx: Tuple[int, int, int]
    _perturb: bool

    _calculated_length: int

    def __init__(self, image_with_division_list: List[_ImageWithDivisions], time_window: Tuple[int, int],
                 patch_shape_zyx: Tuple[int, int, int], perturb: bool):
        self._image_with_divisions_list = image_with_division_list
        self._time_window = time_window
        self._patch_shape_zyx = patch_shape_zyx
        self._perturb = perturb

        # Calculate the length once, to avoid recalculating it every time __len__ is called
        self._calculated_length = sum(len(image.dividing) for image in image_with_division_list)

    def __iter__(self) -> Iterable[Tuple[Tensor, bool]]:
        for image_with_divisions in self._image_with_divisions_list:
            image = image_with_divisions.load_image_time_stack(self._time_window)
            label = image_with_divisions.xyz_positions[:, [2, 1, 0]]
            dividing = image_with_divisions.dividing

            image = normalize(image)

            for crop, division in generate_patches_division(image, label, dividing, self._patch_shape_zyx, self._perturb):
                if self._perturb:
                    crop = apply_noise(crop)
                yield crop, division

    def __len__(self) -> int:
        return self._calculated_length


# Creates training and validation data from an image_with_positions_list
def training_data_creator_from_raw(image_with_divisions_list: List[_ImageWithDivisions], time_window, patch_shape,
                                   batch_size: int, mode, split_proportion: float = 0.8, perturb=True):
    if mode == "train":
        image_with_divisions_list = image_with_divisions_list[:round(split_proportion * len(image_with_divisions_list))]
    elif mode == "validation":
        image_with_divisions_list = image_with_divisions_list[round(split_proportion * len(image_with_divisions_list)):]

    dataset = _TorchDataset(image_with_divisions_list, time_window=time_window, patch_shape_zyx=patch_shape, perturb=perturb)
    dataset = PrefetchingDataset(dataset, buffer_size=100)
    if mode == "train":
        dataset = ShufflingDataset(dataset, buffer_size=batch_size * 100)
    dataset = RepeatingDataset(dataset)
    return DataLoader(dataset, batch_size=batch_size)


# Normalizes image data
def normalize(image):
    image = keras.ops.divide(keras.ops.subtract(image, keras.ops.min(image)), keras.ops.subtract(keras.ops.max(image), keras.ops.min(image)))
    return image


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
        # first, crop to larger region
        init_crop = image_padded[center_point[0]: center_point[0] + patch_shape[0],
                          center_point[1]: center_point[1] + 2*patch_shape[1],
                          center_point[2]: center_point[2] + 2*patch_shape[2], :]

        # apply perturbations
        if perturb:
            #init_crop = apply_random_perturbations_stacked(init_crop)
            random = keras.random.uniform((1,))
            init_crop = keras.ops.cond(random<0.5,
                                lambda: apply_random_flips(init_crop),
                                lambda: apply_random_perturbations_stacked(init_crop))

            #init_crop = black_out(init_crop)

        # second, crop to the center region
        crop = init_crop[:,
               keras.ops.cast(patch_shape[1] / 2, "int32"): keras.ops.cast(patch_shape[1] / 2, "int32") + patch_shape[1],
               keras.ops.cast(patch_shape[2] / 2, "int32"): keras.ops.cast(patch_shape[2] / 2, "int32") + patch_shape[2], :]

        return crop

    # create stack of crops centered around positions
    for single_label, single_division in zip(label, dividing):
        crop = single_patch(single_label)
        yield crop, single_division


def apply_random_perturbations_stacked(stacked):
    image_shape = keras.ops.cast(keras.ops.shape(stacked), "float32")

    transforms = []
    # random rotation in xy
    transform = image_transforms.angles_to_projective_transforms(
        keras.random.uniform([], -np.pi, np.pi), image_shape[1], image_shape[2])
    transforms.append(transform)
    # random scale 80% to 120% size
    scale = keras.random.uniform([], 0.8, 1.2, dtype="float32")
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
    random = keras.random.uniform((1,))
    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[1]), lambda: stacked)

    random = keras.random.uniform((1,))
    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[2]), lambda: stacked)

    random = keras.random.uniform((1,))
    stacked = keras.ops.cond(random<0.5, lambda: keras.ops.flip(stacked, axis=[0]), lambda: stacked)

    return stacked


def apply_noise(image):
    # add noise
    # take power of image to increase or reduce contrast
    image = keras.ops.power(image, keras.random.uniform((1,), minval=0.3, maxval=1.7))

    return image


