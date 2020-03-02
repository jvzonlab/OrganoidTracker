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
"""Code to do the actual training."""
import argparse
import os
from typing import Tuple, List

import tensorflow as tf
import logging

from functools import partial

from organoid_tracker.position_detection_cnn.convolutional_neural_network import build_fcn_model, TRAIN_TFRECORD, TEST_TFRECORD
from organoid_tracker.position_detection_cnn.training_dataset import Dataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


def train(input_dir: str, checkpoint_dir: str, *, patch_size_zyx: Tuple[int, int, int],
          image_size_zyx: Tuple[int, int, int], batch_size: int, use_cpu_output: bool, max_steps: int):
    logger.info('Training network with settings: {}'.format(vars()))

    def input_fn(dataset_path, mode, patch_size_zyx: List[int], image_size_zyx, batch_size, use_cpu):
        dataset = Dataset(dataset_path, batch_size=batch_size, patch_shape=patch_size_zyx, image_size_zyx=image_size_zyx,
                          mode=mode, use_cpu=use_cpu)
        next_features, next_labels = dataset.iterator.get_next()
        data_shape = [batch_size,] + patch_size_zyx + [1, ] if use_cpu else [batch_size, 1] + patch_size_zyx
        next_features.set_shape(data_shape)
        next_labels.set_shape(data_shape)
        return {'data': next_features}, next_labels


    estimator = tf.estimator.Estimator(model_fn=partial(build_fcn_model, use_cpu=use_cpu_output),
                                       model_dir=checkpoint_dir)

    train_spec = tf.estimator.TrainSpec(input_fn=partial(input_fn,
                                                         dataset_path=os.path.join(input_dir, TRAIN_TFRECORD),
                                                         batch_size=batch_size,
                                                         mode=tf.estimator.ModeKeys.TRAIN,
                                                         patch_size_zyx=list(patch_size_zyx),
                                                         image_size_zyx=image_size_zyx,
                                                         use_cpu=use_cpu_output), max_steps=max_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=partial(input_fn,
                                                       dataset_path=os.path.join(input_dir, TEST_TFRECORD),
                                                       batch_size=batch_size,
                                                       mode=tf.estimator.ModeKeys.EVAL,
                                                       patch_size_zyx=list(patch_size_zyx),
                                                       image_size_zyx=image_size_zyx,
                                                       use_cpu=use_cpu_output),
                                      steps=None,
                                      throttle_secs=1800)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

