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
import os
import random
import shutil
from typing import List, Optional, Iterable, Tuple

import numpy
import tensorflow as tf
from numpy import ndarray

from ai_track.core import TimePoint
from ai_track.core.experiment import Experiment
from ai_track.core.images import Images
from ai_track.position_detection_cnn.convolutional_neural_network import TRAIN_TFRECORD, TEST_TFRECORD

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


class _ImageWithPositions:
    _time_point: TimePoint
    _images: Images
    xyz_positions: ndarray
    experiment_name: str

    def __init__(self, experiment_name: str, images: Images, time_point: TimePoint, xyz_positions: ndarray):
        # xyz positions: 2D numpy integer array of cell nucleus positions: [ [x,y,z], [x,y,z], ...]
        self.experiment_name = experiment_name
        self._time_point = time_point
        self._images = images
        self.xyz_positions = xyz_positions

    def load_image(self) -> Optional[ndarray]:
        return self._images.get_image_stack(self._time_point)

    def __str__(self) -> str:
        return f"{self.experiment_name} t{self._time_point.time_point_number()}"


def _create_serialized_data(image_with_positions: _ImageWithPositions, image_size_zyx: Tuple[int, int, int]):
    sub_data_xyz = image_with_positions.xyz_positions

    # int64 is an accepted format for serializing in tfrecord
    multi_im = image_with_positions.load_image()
    if multi_im is None:
        raise Exception()

    # this will create the labels, we create them at this stage
    # it is a preprocessing step, it is less flexible but more efficient for later
    # create a volume the same shape as the images volume,
    # will contain little gaussian disks where the labels are, float32 format values between 0-1
    markers = numpy.zeros(multi_im.shape).astype(numpy.float32)

    # this fills in the labels volume (markers) with the gaussian disks
    for dz in numpy.unique(sub_data_xyz[:, -1]):
        mask = None
        for xyz in sub_data_xyz[numpy.where(sub_data_xyz[:, -1] == dz)]:
            y, x = numpy.ogrid[-xyz[1]:multi_im.shape[1] - xyz[1], -xyz[0]:multi_im.shape[2] - xyz[0]]
            sigma = 2
            gaussian_mask = (x ** 2 + y ** 2 < 4 ** 2) * numpy.exp(-(x ** 2 + y ** 2) /
                                                                   (2. * sigma ** 2))
            if mask is None:
                mask = gaussian_mask
            else:
                mask = numpy.where(gaussian_mask == 0, mask, gaussian_mask)
        if mask is not None:
            if int(dz) < len(markers):  # Skip positions that were set outside the images
                markers[int(dz)] = mask

    # we need all of the volume to have the same shape
    # so pad smaller images with zeroes
    if multi_im.shape[0] > image_size_zyx[0] or multi_im.shape[1] > image_size_zyx[1] or multi_im.shape[2] > image_size_zyx[2]:
        raise Exception(f"Image is bigger than maximum allowed size. Image is (z,y,x) {multi_im.shape}, max size is"
                        f" {image_size_zyx}")

    # pad data and labels in the same way
    data = numpy.zeros(image_size_zyx).astype(numpy.int64)
    dz = int((data.shape[0] - multi_im.shape[0]) / 2)
    dy = int((data.shape[1] - multi_im.shape[1]) / 2)
    dx = int((data.shape[2] - multi_im.shape[2]) / 2)
    data[dz: dz + multi_im.shape[0], dy: dy + multi_im.shape[1], dx: dx + multi_im.shape[2]] = multi_im
    label = numpy.zeros(image_size_zyx).astype(numpy.float32)
    label[dz: dz + multi_im.shape[0], dy: dy + multi_im.shape[1], dx: dx + multi_im.shape[2]] = markers

    # we store data and label with sparse matrix format because the labels are mostly zeros and data also but less
    # this reduces size of tfrecord
    # maybe could be changed to dense matrix for data only depending on the amount of zeroes
    label_non_zero = numpy.where(label > 0.0)
    data_non_zero = numpy.where(data != 0)
    feature = {
        'data_index_0': tf.train.Feature(int64_list=tf.train.Int64List(value=data_non_zero[0])),
        'data_index_1': tf.train.Feature(int64_list=tf.train.Int64List(value=data_non_zero[1])),
        'data_index_2': tf.train.Feature(int64_list=tf.train.Int64List(value=data_non_zero[2])),
        'data_value': tf.train.Feature(int64_list=tf.train.Int64List(value=data[data_non_zero])),
        'label_index_0': tf.train.Feature(int64_list=tf.train.Int64List(value=label_non_zero[0])),
        'label_index_1': tf.train.Feature(int64_list=tf.train.Int64List(value=label_non_zero[1])),
        'label_index_2': tf.train.Feature(int64_list=tf.train.Int64List(value=label_non_zero[2])),
        'label_value': tf.train.Feature(float_list=tf.train.FloatList(value=label[label_non_zero])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    return serialized


def _make_tfrecord(tfrecord_path: str, image_with_positions_list: List[_ImageWithPositions],
                   image_size_zyx: Tuple[int, int, int]):
    if os.path.exists(tfrecord_path):
        os.remove(tfrecord_path)

    tfwriter = tf.python_io.TFRecordWriter(tfrecord_path)

    for i, image_with_positions in enumerate(image_with_positions_list):
        logging.info('processing file {}/{} for {}'.format(i + 1, len(image_with_positions_list), tfrecord_path))

        tfwriter.write(_create_serialized_data(image_with_positions, image_size_zyx))

    tfwriter.close()


def create_training_data(experiments: Iterable[Experiment], *, out_dir: str, split_proportion: float = 0.8,
                         image_size_zyx: Tuple[int, int, int]):
    """
    This script creates the dataset for training in the format tfrecord,
    from images and corresponding annotations (json files)
    output : train.tfrecord and test.tfrecord
    """

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)  # This empties the directory, but actually removing
                                # it may not happen immediately on Windows
    os.makedirs(out_dir, exist_ok=True)

    image_with_positions_list = []
    for experiment in experiments:
        # read a complete experiment

        for time_point in experiment.positions.time_points():
            # read a single time point
            positions = experiment.positions.of_time_point(time_point)

            # read positions to numpy array
            positions_xyz = list()
            for position in positions:
                positions_xyz.append([position.x, position.y, position.z])
            positions_xyz = numpy.array(positions_xyz, dtype=numpy.int32)

            image_with_positions_list.append(
                _ImageWithPositions(str(experiment.name), experiment.images, time_point, positions_xyz))

    # shuffle images & positions pseudo-randomly and then split into test and training set
    random.seed("using a fixed seed to ensure reproducibility")
    random.shuffle(image_with_positions_list)
    train_eval_split = int(split_proportion * len(image_with_positions_list))
    train_files = image_with_positions_list[:train_eval_split]
    test_files = image_with_positions_list[train_eval_split:]
    numpy.savetxt(os.path.join(out_dir, 'train_files.txt'), [str(train_file) for train_file in train_files], fmt="%s")
    numpy.savetxt(os.path.join(out_dir, 'test_files.txt'), [str(test_file) for test_file in test_files], fmt="%s")

    _make_tfrecord(os.path.join(out_dir, TRAIN_TFRECORD), train_files, image_size_zyx)
    _make_tfrecord(os.path.join(out_dir, TEST_TFRECORD), test_files, image_size_zyx)
