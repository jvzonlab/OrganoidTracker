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
from torch.utils.data import IterableDataset, DataLoader

from organoid_tracker.neural_network import Tensor
from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import ImageWithPositions


class _TorchDataset(IterableDataset):
    _image_with_divisions_list: List[ImageWithPositions]
    _time_window: Tuple[int, int]
    _patch_shape_zyx: Tuple[int, int, int]

    _calculated_length: int

    def __init__(self, image_with_division_list: List[ImageWithPositions], time_window: Tuple[int, int],
                 patch_shape_zyx: Tuple[int, int, int]):
        self._image_with_divisions_list = image_with_division_list
        self._time_window = time_window
        self._patch_shape_zyx = patch_shape_zyx

        # Calculate the length once, to avoid recalculating it every time __len__ is called
        self._calculated_length = sum(len(image.xyz_positions) for image in image_with_division_list)

    def __iter__(self) -> Iterable[Tensor]:
        for image_with_divisions in self._image_with_divisions_list:
            image = image_with_divisions.load_image_time_stack(self._time_window)
            label = image_with_divisions.xyz_positions[:, [2, 1, 0]]

            image = normalize(image)

            yield from generate_patches_division(image, label, self._patch_shape_zyx)

    def __len__(self) -> int:
        return self._calculated_length


# Creates training and validation data from an image_with_positions_list
def prediction_data_creator(image_with_positions_list: List[ImageWithPositions], time_window, patch_shape):
    dataset = _TorchDataset(image_with_positions_list, time_window, patch_shape)
    return DataLoader(dataset, batch_size=1)


# Normalizes image data
def normalize(image):
    image = keras.ops.divide(keras.ops.subtract(image, keras.ops.min(image)),
                             keras.ops.subtract(keras.ops.max(image), keras.ops.min(image)))
    return image


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
    for single_label in label:
        crop = single_patch(single_label)
        yield crop
