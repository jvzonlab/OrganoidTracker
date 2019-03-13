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

import logging
import math
import os
from functools import partial
from typing import Optional, Tuple
from numpy import ndarray

import numpy
import tensorflow as tf
from skimage.feature import peak_local_max
from tifffile import tifffile

from autotrack.core import TimePoint
from autotrack.core.images import Images
from autotrack.core.position import Position
from autotrack.core.position_collection import PositionCollection
from autotrack.core.resolution import ImageResolution
from autotrack.position_detection_cnn.convolutional_neural_network import build_fcn_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


def _get_slices(volume_zyx: Tuple[int, int, int], slice_size_zyx: Tuple[int, int, int], slice_margin_zyx: Tuple[int, int, int]):
    start_x, start_y, start_z = 0, 0, 0
    while start_z < volume_zyx[0]:
        while start_y < volume_zyx[1]:
            while start_x < volume_zyx[2]:
                yield ((start_z - slice_margin_zyx[0], start_z + slice_size_zyx[0] + slice_margin_zyx[0]),
                       (start_y - slice_margin_zyx[1], start_y + slice_size_zyx[1] + slice_margin_zyx[1]),
                       (start_x - slice_margin_zyx[2], start_x + slice_size_zyx[0] + slice_margin_zyx[2]))
                start_x += slice_size_zyx[2]
            start_y += slice_size_zyx[1]
        start_z += slice_size_zyx[0]


def _reconstruct_volume(multi_im: ndarray, resolution: ImageResolution) -> Tuple[ndarray, int]:
    """Reconstructs a volume so that the xy scale is roughly equal to the z scale. Returns the used scale."""
    # Make sure that
    mid_layers_nb = int(resolution.pixel_size_z_um / resolution.pixel_size_x_um) - 1
    if mid_layers_nb < 0:
        raise ValueError("x resolution is courser than z resolution")
    if mid_layers_nb == 0:
        return multi_im, 1  # No need to reconstruct anything

    out_img = numpy.zeros(
        (int(len(multi_im) + mid_layers_nb * (len(multi_im) - 1) + 2 * mid_layers_nb),) + multi_im[0].shape).astype(
        multi_im[0].dtype)

    layer_index = mid_layers_nb + 1
    orig_index = []

    for i in range(len(multi_im) - 1):

        for layer in range(mid_layers_nb + 1):
            t = float(layer) / (mid_layers_nb + 1)
            interpolate = ((1 - t) * (multi_im[i]).astype(float) + t * (multi_im[i + 1]).astype(float))

            out_img[layer_index] = interpolate

            if t == 0:
                orig_index.append(layer_index)
            layer_index += 1

    return out_img, mid_layers_nb + 1


def _next_multiple_of_32(number: int) -> int:
    """31 becomes 32, 32 stays 32, 33 becomes 64, and so on."""
    return math.ceil(number / 32) * 32


def _input_fn(images: Images, split: bool):
    image_size_zyx = images.image_loader().get_image_size_zyx()
    image_size_x = image_size_zyx[2]
    image_size_z = image_size_zyx[0]

    def gen_images():
        for time_point in images.time_points():
            image_data = numpy.array(images.get_image(time_point).array).astype(numpy.float32)
            image_data = (image_data - numpy.min(image_data)) / (numpy.max(image_data) - numpy.min(image_data))
            data = numpy.zeros((_next_multiple_of_32(image_size_z), image_size_x, image_size_x)).astype(numpy.float32)
            z = int((data.shape[0] - image_data.shape[0]) / 2)
            data[z: z + image_data.shape[0], :, :] = image_data

            if split:
                part_xy_size = image_size_x // 2 + 64
                yield {'data': data[numpy.newaxis, numpy.newaxis, :, :part_xy_size, :part_xy_size]}  # Top left
                yield {'data': data[numpy.newaxis, numpy.newaxis, :, :part_xy_size, -part_xy_size:]}  # Top right
                yield {'data': data[numpy.newaxis, numpy.newaxis, :, -part_xy_size:, :part_xy_size]}  # Bottom left
                yield {'data': data[numpy.newaxis, numpy.newaxis, :, -part_xy_size:, -part_xy_size:]}  # Bottom right
            else:
                yield {'data': data[numpy.newaxis, numpy.newaxis, :, :, :]}

    if split_in_four:
        part_xy_size = image_size_x // 2 + 64
        output_shape = [1, 1, _next_multiple_of_32(image_size_z), part_xy_size, part_xy_size]
    else:
        output_shape = [1, 1, _next_multiple_of_32(image_size_z), image_size_x, image_size_x]
    dataset = tf.data.Dataset.from_generator(gen_images,
                                             output_types={'data': tf.float32},
                                             output_shapes={'data': tf.TensorShape(output_shape)})
    iterator = dataset.make_one_shot_iterator()
    next_features = iterator.get_next()
    return next_features


def predict(images: Images, checkpoint_dir: str, out_dir: Optional[str] = None, split_in_four: bool = False
            ) -> PositionCollection:
    min_time_point_number = images.image_loader().first_time_point_number()
    if min_time_point_number is None:
        raise ValueError("No images were loaded")
    image_size_zyx = images.image_loader().get_image_size_zyx()
    image_size_x = image_size_zyx[2]
    image_size_z = image_size_zyx[0]
    output_size_z = _next_multiple_of_32(image_size_z)
    output_offset_z = (output_size_z - image_size_z) // 2
    resolution = images.resolution()

    estimator = tf.estimator.Estimator(model_fn=partial(build_fcn_model, use_cpu=False), model_dir=checkpoint_dir)
    predictions = estimator.predict(input_fn=lambda: _input_fn(images, split_in_four))

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    half_x_size = image_size_x // 2
    complete_prediction = numpy.empty((output_size_z, image_size_x, image_size_x),
                                      dtype=numpy.float32) if split_in_four else None
    all_positions = PositionCollection()

    for index, p in enumerate(predictions):

        # Remove batch and channel dimensions, shape should be (z, y, x)
        prediction = numpy.squeeze(p['prediction'])

        # If the image was split: reconstruct the larger image from the four parts
        if split_in_four:
            sub_index = index % 4
            if sub_index == 0:
                complete_prediction[:, :half_x_size, :half_x_size] = prediction[:, :half_x_size, :half_x_size]
                continue
            elif sub_index == 1:
                complete_prediction[:, :half_x_size, -half_x_size:] = prediction[:, :half_x_size, -half_x_size:]
                continue
            elif sub_index == 2:
                complete_prediction[:, -half_x_size:, :half_x_size] = prediction[:, -half_x_size:, :half_x_size]
                continue

            # Prediction is now completed
            complete_prediction[:, -half_x_size:, -half_x_size:] = prediction[:, -half_x_size:, -half_x_size:]
            prediction = complete_prediction
            image_index = index // 4
        else:
            image_index = index

        time_point = TimePoint(min_time_point_number + image_index)
        print("Working on time point", time_point.time_point_number(), "...")
        image_offset = images.offsets.of_time_point(time_point)

        if out_dir is not None:
            image_name = "image-" + str(time_point.time_point_number())
            tifffile.imsave(os.path.join(out_dir, '{}.tif'.format(image_name)), prediction)
        im, z_divisor = _reconstruct_volume(prediction, resolution) # interpolate between layers for peak detection

        #can do the same thing with data to visualize
        # imsource, _ = _reconstruct_volume(numpy.squeeze(p['data']))

        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(im, min_distance=round(3.2 / resolution.pixel_size_x_um), threshold_abs=0.1)
        for coordinate in coordinates:
            pos = Position(coordinate[2], coordinate[1], coordinate[0] / z_divisor - output_offset_z,
                           time_point=time_point) + image_offset
            all_positions.add(pos)
    return all_positions