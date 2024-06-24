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
from typing import List, Iterable, Dict, Tuple

from functools import partial

import keras
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

from organoid_tracker.neural_network import image_transforms, Tensor
from organoid_tracker.neural_network.link_detection_cnn.ImageWithLinks_to_tensor_loader import load_images_with_links
from organoid_tracker.neural_network.link_detection_cnn.training_data_creator import _ImageWithLinks


# Creates training and validation data from an image_with_positions_list
from organoid_tracker.neural_network.link_detection_cnn.training_dataset import normalize, _add_3d_coord


class _TorchDataset(IterableDataset):
    _image_with_links_list: List[_ImageWithLinks]
    _time_window: Tuple[int, int]
    _patch_shape_zyx: Tuple[int, int, int]

    _calculated_length: int

    def __init__(self, image_with_links_list: List[_ImageWithLinks], time_window: Tuple[int, int],
                 patch_shape_zyx: Tuple[int, int, int]):
        self._image_with_links_list = image_with_links_list
        self._time_window = time_window
        self._patch_shape_zyx = patch_shape_zyx

        # Calculate the length once, to avoid recalculating it every time __len__ is called
        self._calculated_length = sum(len(image.xyz_positions) for image in image_with_links_list)

    def __iter__(self) -> Iterable[Dict[str, Tensor]]:
        for image_with_divisions in self._image_with_links_list:
            image, target_image, label, target_label, distances, linked = load_images_with_links(image_with_divisions, self._time_window)
            image = image_with_divisions.load_image_time_stack(self._time_window)
            label = image_with_divisions.xyz_positions[:, [2, 1, 0]]

            image = normalize(image)
            target_image = normalize(target_image)

            for stacked_crop, stacked_targed_crop, distances in generate_patches_links(image, target_image, label, target_label, distances, self._patch_shape_zyx):
                stacked_crop = _add_3d_coord(stacked_crop, distances)
                stacked_targed_crop = _add_3d_coord(stacked_targed_crop, distances, reversable=True)
                yield {'input_1': stacked_crop, 'input_2': stacked_targed_crop, 'input_distances': distances}

    def __len__(self) -> int:
        return self._calculated_length


def prediction_data_creator(load_images_with_links_list: List[_ImageWithLinks], time_window: Tuple[int, int],
                            patch_shape_zyx: Tuple[int, int, int]):
    return DataLoader(_TorchDataset(load_images_with_links_list, time_window, patch_shape_zyx), batch_size=50)


def generate_patches_links(image, target_image, label, target_label, distances, patch_shape):
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
        yield stacked_crop, stacked_target_crop, distance


def add_3d_coord(image, target_image, distances):
    image = _add_3d_coord(image, distances)

    return image, _add_3d_coord(target_image, distances, reversable=True), distances
