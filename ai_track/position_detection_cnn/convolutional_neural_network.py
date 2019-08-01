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


TRAIN_TFRECORD = "train.tfrecord"  # Data file used for model training
TEST_TFRECORD = "test.tfrecord"  # Data file used for model evaluation


import tensorflow as tf


def build_fcn_model(features, labels, mode, use_cpu=False):
    data_format = _to_data_format(use_cpu)

    layer = features['data']

    to_concat = []

    # classical fully convolutional UNET with one change (coordconv layers)
    # to test: batch normalization, residual connections etc

    layer, to_concat_layer = conv_block(2, layer, 16, use_cpu, 'down_block1')
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(2, layer, 64, use_cpu, 'down_block2')
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(2, layer, 128, use_cpu, 'down_block3')
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(2, layer, 256, use_cpu, 'down_block4')
    to_concat.append(to_concat_layer)

    layer = deconv_block(2, layer, to_concat.pop(), 256, use_cpu, 'up_block1')
    layer = deconv_block(2, layer, to_concat.pop(), 128, use_cpu, 'up_block2')
    layer = deconv_block(2, layer, to_concat.pop(), 64, use_cpu, 'up_block3')
    layer = deconv_block(2, layer, to_concat.pop(), 16, use_cpu, 'up_block4')

    layer = tf.layers.conv3d(inputs=layer, filters=1,
                             kernel_size=[3, 3, 3], padding="same",
                             data_format=data_format,
                             name='out_conv')
    predictions = {
        'prediction': layer,
        'data': features['data']
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = add_loss(layer, labels)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def add_loss(predictions, labels):
    with tf.name_scope('loss'):
        non_zero = tf.cast(tf.count_nonzero(labels), tf.float32)
        full_size = tf.cast(tf.size(labels), tf.float32)
        # weight the loss by the amount of non zeroes values in label
        fraction_non_zero = tf.divide(non_zero, full_size)
        fraction_zero = tf.subtract(1., fraction_non_zero)

        weights = tf.where(tf.equal(labels, 0),
                           tf.fill(tf.shape(labels), fraction_non_zero),
                           tf.multiply(labels, fraction_zero))

        return tf.losses.mean_squared_error(labels, predictions, weights)


def conv_block(n_conv, layer, filters, use_cpu, name):
    data_format = _to_data_format(use_cpu)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for index in range(n_conv):
            layer = conv_coord_3d(layer, filters, index,
                                  use_cpu)  # To test : is coordconv needed in all layers or just first?
        to_concat = layer
        layer = tf.layers.max_pooling3d(inputs=layer, pool_size=2, strides=2, padding='same',
                                        name='pool', data_format=data_format)
        return layer, to_concat


def deconv_block(n_conv, layer, to_concat_layer, filters, use_cpu, name):
    data_format = _to_data_format(use_cpu)
    axis = 1
    if use_cpu:
        axis = -1

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        layer = tf.layers.conv3d_transpose(layer, filters, [3, 3, 3], padding='same',
                                           strides=(2, 2, 2), data_format=data_format,
                                           name='upconv')
        layer = tf.concat([layer, to_concat_layer], axis=axis)
        for index in range(n_conv):
            layer = conv_coord_3d(layer, filters, index, use_cpu)

        return layer


# 3D convolution replaced by CoordConv
# An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
# https://arxiv.org/abs/1807.03247

def add_3d_coord(layer):
    # FIXME can we make this using a loop?
    im_shape = tf.shape(layer)[2:]
    batch_size_tensor = tf.shape(layer)[0]

    ##0 dim (X)
    xval_range = tf.range(im_shape[1])
    xval_range = tf.expand_dims(xval_range, 0)
    ones = tf.ones([im_shape[1], 1], dtype=tf.int32)
    xval_range = ones * xval_range
    ones = tf.ones([im_shape[0], 1, 1], dtype=tf.int32)
    xval_range = ones * xval_range
    xval_range = tf.ones([batch_size_tensor, 1, 1, 1, 1], dtype=tf.int32) * xval_range
    xval_range = tf.cast(xval_range, 'float32') / (tf.cast(im_shape[1], 'float32') - 1)
    xval_range = xval_range * 2 - 1

    ##1 dim (Y)
    yval_range = tf.range(im_shape[2])
    yval_range = tf.expand_dims(yval_range, 1)
    ones = tf.ones([1, im_shape[2]], dtype=tf.int32)
    yval_range = ones * yval_range
    ones = tf.ones([im_shape[0], 1, 1], dtype=tf.int32)
    yval_range = ones * yval_range
    yval_range = tf.ones([batch_size_tensor, 1, 1, 1, 1], dtype=tf.int32) * yval_range
    yval_range = tf.cast(yval_range, 'float32') / (tf.cast(im_shape[2], 'float32') - 1)
    yval_range = yval_range * 2 - 1

    ##2 dim (Z)
    zval_range = tf.range(im_shape[0])
    zval_range = tf.expand_dims(tf.expand_dims(zval_range, 1), 1)
    ones = tf.ones([1, im_shape[1], im_shape[2]], dtype=tf.int32)
    zval_range = ones * zval_range
    zval_range = tf.ones([batch_size_tensor, 1, 1, 1, 1], dtype=tf.int32) * zval_range
    zval_range = tf.cast(zval_range, 'float32') / (tf.cast(im_shape[0], 'float32') - 1)
    zval_range = zval_range * 2 - 1

    ret = tf.concat([layer, xval_range, yval_range, zval_range], axis=1)
    return ret


def conv_coord_3d(layer, filters, index, use_cpu):
    if not use_cpu:
        layer = add_3d_coord(layer)  # Not yet supported for use_cpu

    data_format = "channels_last" if use_cpu else "channels_first"
    layer = tf.layers.conv3d(inputs=layer, filters=filters,
                             kernel_size=[3, 3, 3], padding="same",
                             activation=tf.nn.relu, data_format=data_format,
                             name='conv{0}'.format(index + 1))
    return layer


def _to_data_format(use_cpu):
    """Returns the optimal data format for either the GPU or the CPU."""
    if use_cpu:
        return "channels_last"
    else:
        return "channels_first"
