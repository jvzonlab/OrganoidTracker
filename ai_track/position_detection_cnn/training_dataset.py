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
from typing import Tuple

import tensorflow as tf
from functools import partial
import numpy as np


class Dataset:

    def __init__(self, filename: str, *, batch_size: int, patch_shape, image_size_zyx: Tuple[int, int, int],
                 mode: str, use_cpu: bool = False):
        self.use_cpu = use_cpu

        dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=32)
        dataset = dataset.map(partial(Dataset.parse_tfrecord, image_size_zyx=image_size_zyx))

        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10)  # small shuffling so that each iteration is a little different
            dataset = dataset.repeat()

        dataset = dataset.batch(1) #generate patches will make the batch
        dataset = dataset.flat_map(partial(self.generate_patches,
                                           batch_size=batch_size,
                                           patch_shape=patch_shape,
                                           mode=mode))
        dataset = dataset.prefetch(2)

        self.iterator = dataset.make_one_shot_iterator()

    @staticmethod
    def parse_tfrecord(example, image_size_zyx: Tuple[int, int, int]):
        features = {'label': tf.SparseFeature(index_key=['label_index_0', 'label_index_1', 'label_index_2'],
                                              value_key='label_value',
                                              dtype=tf.float32,
                                              size=image_size_zyx),
                    'data': tf.SparseFeature(index_key=['data_index_0', 'data_index_1', 'data_index_2'],
                                             value_key='data_value',
                                             dtype=tf.int64,
                                             size=image_size_zyx)
                    }
        parsed_features = tf.parse_single_example(example, features)
        return parsed_features

    @staticmethod
    def convert_ncwh_to_nhwc(image_data, label):
        return image_data[0][:, :, :, tf.newaxis], label[0][:, :, :, tf.newaxis]

    def apply_random_perturbations(self, image_data, label, patch_shape):
        # Note: this function automatically converts the given data to CPU output format if self.use_cpu is True
        with tf.name_scope('augmentation'):
            random_flip_x = tf.less(tf.random_uniform(()), 0.5)
            image_data, label = tf.cond(random_flip_x,
                                        lambda: (tf.reverse(image_data, [3]), tf.reverse(label, [3])),
                                        lambda: (image_data, label))

            random_flip_y = tf.less(tf.random_uniform(()), 0.5)
            image_data, label = tf.cond(random_flip_y,
                                        lambda: (tf.reverse(image_data, [2]), tf.reverse(label, [2])),
                                        lambda: (image_data, label))

            random_flip_z = tf.less(tf.random_uniform(()), 0.5)
            image_data, label = tf.cond(random_flip_z,
                                        lambda: (tf.reverse(image_data, [1]), tf.reverse(label, [1])),
                                        lambda: (image_data, label))

            #TODO small translations in Z only

            #convert to NHWC for tf.image functions
            image_data, label = self.convert_ncwh_to_nhwc(image_data, label)

            transforms = []
            # random rotation 0-360
            transform = tf.contrib.image.angles_to_projective_transforms(
                tf.random_uniform([], -np.pi, np.pi), patch_shape[1], patch_shape[2])
            transforms.append(transform)
            # random scale 80% to 120% size
            scale = tf.random_uniform([], 0.8, 1.2, dtype=tf.float32)
            transform = tf.convert_to_tensor([[1 / scale, 0., patch_shape[1] / 2 / scale * (scale - 1),
                                               0., 1 / scale, patch_shape[2] / 2 / scale * (scale - 1), 0.,
                                               0.]], dtype=tf.float32)
            transforms.append(transform)

            compose_transforms = tf.contrib.image.compose_transforms(*transforms)
            # compose rotation scale
            image_data = tf.contrib.image.transform(image_data, compose_transforms, interpolation='BILINEAR')
            label = tf.contrib.image.transform(label, compose_transforms, interpolation='BILINEAR')

            # random brightness and contrast
            #FIXME maybe can replace this with tf.image.per_image_standardization
            image_data = tf.image.random_brightness(image_data, max_delta=0.1)
            image_data = tf.image.random_contrast(image_data, 0.5, 1.5)

            if not self.use_cpu:
                # convert back to NCWH for training
                image_data = image_data[:, :, :, 0][tf.newaxis, :, :, :]
                label = label[:, :, :, 0][tf.newaxis, :, :, :]

            return image_data, label

    def generate_patches(self, features, batch_size, patch_shape, mode):
        #make small patches from the big image, centered on the labels
        #each batch is composed of small patches from ONE image (is it ok?)

        label_sparse = features['label']
        data_sparse = features['data']

        data_dense_shape = label_sparse.dense_shape[1:]
        tf_patch_shape = tf.constant(patch_shape, dtype=tf.int64)
        half_patch_shape = tf.cast(tf.divide(tf_patch_shape, 2), dtype=tf.int64)

        min_patch_corner = tf.constant([0, 0, 0], dtype=tf.int64)
        max_patch_corner = tf.subtract(tf.subtract(data_dense_shape, 1), tf_patch_shape)
        max_patch_corner = tf.maximum(min_patch_corner, max_patch_corner)

        label_indices = tf.expand_dims(label_sparse.indices[:, 1:], axis=0)
        max_indices = tf.reduce_max(label_indices, axis=[1])
        min_indices = tf.reduce_min(label_indices, axis=[1])

        min_indices_corner = tf.subtract(min_indices, half_patch_shape)
        max_indices_corner = tf.subtract(max_indices, half_patch_shape)
        min_indices_bounded = tf.minimum(tf.maximum(min_indices_corner, min_patch_corner), max_patch_corner)
        max_indices_bounded = tf.minimum(tf.maximum(max_indices_corner, min_patch_corner), max_patch_corner)

        label_dense = tf.sparse_tensor_to_dense(label_sparse)
        data_dense = tf.sparse_tensor_to_dense(data_sparse)
        data_dense = tf.cast(data_dense, tf.float32)
        data_dense = tf.div(tf.subtract(data_dense, tf.reduce_min(data_dense)),
                            tf.subtract(tf.reduce_max(data_dense), tf.reduce_min(data_dense)))

        def make_random(min_val, max_val):
            rand_val = tf.cond(tf.not_equal(min_val, max_val),
                               lambda: tf.random_uniform((1,), minval=min_val, maxval=max_val, dtype=tf.int64)[0],
                               lambda: tf.identity(min_val))
            return rand_val

        def make_random_patch():
            rand_val_0 = make_random(min_indices_bounded[:, 0][0], max_indices_bounded[:, 0][0])
            rand_val_1 = make_random(min_indices_bounded[:, 1][0], max_indices_bounded[:, 1][0])
            rand_val_2 = make_random(min_indices_bounded[:, 2][0], max_indices_bounded[:, 2][0])

            rand_corner_start = tf.stack([rand_val_0, rand_val_1, rand_val_2])
            rand_corner_end = tf.add(rand_corner_start, tf_patch_shape)
            sub_label = label_dense[:, rand_corner_start[0]:rand_corner_end[0],
                        rand_corner_start[1]:rand_corner_end[1],
                        rand_corner_start[2]:rand_corner_end[2]]
            sub_data = data_dense[:, rand_corner_start[0]:rand_corner_end[0],
                       rand_corner_start[1]:rand_corner_end[1],
                       rand_corner_start[2]:rand_corner_end[2]]
            return sub_data, sub_label

        all_data = []
        all_label = []

        for i in range(batch_size):
            sub_data, sub_label = make_random_patch()

            if mode == tf.estimator.ModeKeys.TRAIN:
                # Note: the following function automatically converts to CPU output format if self.use_cpu is True
                sub_data, sub_label = self.apply_random_perturbations(sub_data, sub_label, patch_shape)
            elif self.use_cpu:
                sub_data, sub_label = self.convert_ncwh_to_nhwc(sub_data, sub_label)

            all_data.append(sub_data)
            all_label.append(sub_label)

        all_data = tf.stack(all_data)
        all_label = tf.stack(all_label)

        dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(all_data),
                                       tf.data.Dataset.from_tensor_slices(all_label)))
        dataset = dataset.batch(batch_size)
        return dataset

