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
import itertools
import logging
import math
import os
from functools import partial
from typing import Optional, Tuple, Iterable, List, TypeVar
from numpy import ndarray

import numpy
import tensorflow as tf
from skimage.feature import peak_local_max
from tifffile import tifffile

from organoid_tracker.core import TimePoint
from organoid_tracker.core.images import Images
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.imaging import image_slicer
from organoid_tracker.imaging.image_slicer import Slicer3d
from organoid_tracker.position_detection_cnn.convolutional_neural_network import build_fcn_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


_IMAGE_PART_SIZE = (28, 256, 256)  # ZYX size of the image if split is True
_IMAGE_PART_MARGIN = (2, 32, 32)  # Margin inside the image part
_IMAGE_PART_SIZE_PLUS_MARGIN = (_IMAGE_PART_SIZE[0] + 2 * _IMAGE_PART_MARGIN[0],
                                _IMAGE_PART_SIZE[1] + 2 * _IMAGE_PART_MARGIN[1],
                                _IMAGE_PART_SIZE[2] + 2 * _IMAGE_PART_MARGIN[2])


def _get_slices(volume: Tuple[int, int, int]) -> Iterable[Slicer3d]:
    return image_slicer.get_slices(volume, _IMAGE_PART_SIZE, _IMAGE_PART_MARGIN)


T = TypeVar('T')
def _cycle(list: List[T]) -> Iterable[T]:
    """Generator that returns the elements in the list ad infinitum."""
    while True:
        for elem in list:
            yield elem


def _reconstruct_volume(multi_im: ndarray, mid_layers_nb: int) -> Tuple[ndarray, int]:
    """Reconstructs a volume so that the xy scale is roughly equal to the z scale. Returns the used scale."""
    # Make sure that
    if mid_layers_nb < 0:
        raise ValueError("negative number of mid layers")
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
    image_size_y = image_size_zyx[1]
    image_size_z = image_size_zyx[0]
    image_size_x = max(image_size_x, image_size_y)
    image_size_y = image_size_x

    def gen_images():
        for time_point in images.time_points():
            image_data = numpy.array(images.get_image(time_point).array).astype(numpy.float32)
            image_data = (image_data - numpy.min(image_data)) / (numpy.max(image_data) - numpy.min(image_data))
            data = numpy.zeros((_next_multiple_of_32(image_size_z), _next_multiple_of_32(image_size_y),
                                _next_multiple_of_32(image_size_x))).astype(numpy.float32)
            z = int((data.shape[0] - image_data.shape[0]) / 2)
            data[z: z + image_data.shape[0], 0:image_data.shape[1], 0:image_data.shape[2]] = image_data

            if split:
                for slice in _get_slices(data.shape):
                    slice_data = slice.slice(data)
                    yield {'data': slice_data[numpy.newaxis, numpy.newaxis, :, :, :]}
            else:
                yield {'data': data[numpy.newaxis, numpy.newaxis, :, :, :]}

    if split:
        output_shape = [1, 1, _IMAGE_PART_SIZE_PLUS_MARGIN[0], _IMAGE_PART_SIZE_PLUS_MARGIN[1], _IMAGE_PART_SIZE_PLUS_MARGIN[2]]
    else:
        output_shape = [1, 1, _next_multiple_of_32(image_size_z), _next_multiple_of_32(image_size_y), _next_multiple_of_32(image_size_x)]
    dataset = tf.data.Dataset.from_generator(gen_images,
                                             output_types={'data': tf.float32},
                                             output_shapes={'data': tf.TensorShape(output_shape)})
    iterator = dataset.make_one_shot_iterator()
    next_features = iterator.get_next()
    return next_features


def predict(images: Images, checkpoint_dir: str, out_dir: Optional[str] = None, split: bool = False,
            mid_layers_nb: int = 5, min_peak_distance_px: int = 9) -> PositionCollection:
    min_time_point_number = images.image_loader().first_time_point_number()
    if min_time_point_number is None:
        raise ValueError("No images were loaded")
    image_size_zyx = images.image_loader().get_image_size_zyx()
    image_size_x = image_size_zyx[2]
    image_size_y = image_size_zyx[1]
    image_size_z = image_size_zyx[0]
    output_size_x = _next_multiple_of_32(image_size_x)
    output_size_y = _next_multiple_of_32(image_size_y)
    output_size_z = _next_multiple_of_32(image_size_z)
    output_offset_z = math.ceil((output_size_z - image_size_z) / 2)

    output_size_x = max(output_size_x, output_size_y)  # Make image a square
    output_size_y = output_size_x

    # Try to turn off slit
    if split and output_size_x <= _IMAGE_PART_SIZE_PLUS_MARGIN[2] and output_size_y <= _IMAGE_PART_SIZE_PLUS_MARGIN[1] \
            and output_size_x <= _IMAGE_PART_SIZE_PLUS_MARGIN[1]:
        split = False  # Input image is so small that it doesn't need to be split

    estimator = tf.estimator.Estimator(model_fn=partial(build_fcn_model, use_cpu=False), model_dir=checkpoint_dir)
    predictions = estimator.predict(input_fn=lambda: _input_fn(images, split))

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    all_positions = PositionCollection()

    slices = list(_get_slices((output_size_z, output_size_y, output_size_x))) if split else []
    complete_prediction = numpy.empty((output_size_z, output_size_y, output_size_x), dtype=numpy.float32)\
        if split else None

    for index, p in enumerate(predictions):

        # Remove batch and channel dimensions, shape should be (z, y, x)
        prediction = numpy.squeeze(p['prediction'])

        # If the image was split: reconstruct the larger image from the four parts
        if split:
            sub_index = index % len(slices)
            slicer = slices[sub_index]
            slicer.place_slice_in_volume(prediction, complete_prediction)

            if sub_index < len(slices) - 1:
                continue  # More subimages to add

            # Prediction is now completed
            prediction = complete_prediction
            image_index = index // len(slices)
        else:
            image_index = index

        time_point = TimePoint(min_time_point_number + image_index)
        print("Working on time point", time_point.time_point_number(), "...")
        image_offset = images.offsets.of_time_point(time_point)

        prediction = prediction[output_offset_z : output_offset_z + image_size_z]
        if out_dir is not None:
            image_name = "image_" + str(time_point.time_point_number())
            tifffile.imsave(os.path.join(out_dir, '{}.tif'.format(image_name)), prediction, compress=9)
        im, z_divisor = _reconstruct_volume(prediction, mid_layers_nb) # interpolate between layers for peak detection

        #can do the same thing with data to visualize
        # imsource, _ = _reconstruct_volume(numpy.squeeze(p['data']))

        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(im, min_distance=min_peak_distance_px, threshold_abs=0.1, exclude_border=False)
        for coordinate in coordinates:
            pos = Position(coordinate[2], coordinate[1], coordinate[0] / z_divisor - 1,
                           time_point=time_point) + image_offset
            all_positions.add(pos)
    return all_positions
