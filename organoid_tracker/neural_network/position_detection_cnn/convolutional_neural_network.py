from typing import Tuple

import keras
import keras.callbacks

from organoid_tracker.neural_network.position_detection_cnn.custom_filters import blur_labels
from organoid_tracker.neural_network.position_detection_cnn.loss_functions import position_recall, position_precision, \
    overcount, loss


def build_model(shape: Tuple, batch_size):
    # Input layer
    input = keras.Input(shape=shape, batch_size=batch_size)

    # Add coordinates
    layer = input #add_3d_coord(input, only_z = True)

    # convolutions
    to_concat = []

    #filter_sizes = [16, 32, 64, 128, 256]
    filter_sizes = [2, 16, 64, 128, 256]
    #filter_sizes = [2, 8, 16, 32, 64]
    n=2
    layer, to_concat_layer = conv_block(n, layer, filters=filter_sizes[1], kernel=(1, 3, 3), pool_size=(1, 2, 2),
                                        pool_strides=(1, 2, 2), name="down1")#, depth_wise= filter_sizes[0])
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(n, layer, filters=filter_sizes[2], name="down2")#, depth_wise= filter_sizes[1])
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(n, layer, filters=filter_sizes[3], name="down3")#, depth_wise= filter_sizes[2])
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(n, layer, filters=filter_sizes[4], name="down4")#, depth_wise= filter_sizes[3])
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(n, layer, filters=filter_sizes[4], name="down4A")#, depth_wise= filter_sizes[3])
    to_concat.append(to_concat_layer)

    layer = deconv_block(n, layer, to_concat.pop(), filters=filter_sizes[4], name="up1A")#, depth_wise=True)
    layer = deconv_block(n, layer, to_concat.pop(), filters=filter_sizes[4], name="up1")#, depth_wise=True)
    layer = deconv_block(n, layer, to_concat.pop(), filters=filter_sizes[3], name="up2")#, depth_wise=True)
    layer = deconv_block(n, layer, to_concat.pop(), filters=filter_sizes[2], name="up3")#, depth_wise=True)
    layer = deconv_block(n, layer, to_concat.pop(), filters=filter_sizes[1], kernel=(3, 3, 3), strides=(1, 2, 2), dropout=False, name="up4")
    layer = deconv_block(n, layer, None, filters=filter_sizes[1], kernel=(3, 3, 3), strides=(1, 1, 1),
                       dropout=False, name="up_z")

    # apply final batch_normalization
    layer = keras.layers.BatchNormalization()(layer)

    output = keras.layers.Conv3D(filters=1, kernel_size=3, padding="same", activation='relu', name='out_conv')(layer)

    # blur predictions (leads to less noise-induced peaks) This helps sometimes (?)
    output = blur_labels(output, sigma=1.5, kernel_size=4,  depth=1, normalize=False)
    #output = blur_labels(output, sigma=3, kernel_size=7, depth=3, normalize=False)

    model = keras.Model(inputs=input, outputs=output, name="YOLO")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss=loss, metrics=[position_recall, position_precision, overcount])

    return model


def conv_block(n_conv, layer, filters, kernel=3, pool_size=2, pool_strides=2, dropout=False, name=None, depth_wise= None):
    for index in range(n_conv):

        if depth_wise is not None:
            layer = keras.layers.Conv3D(filters=filters, kernel_size=kernel, groups=depth_wise, padding='same', activation='linear',
                                           name=name + '>vol_conv{0}'.format(index + 1))(layer)
            layer = keras.layers.Conv3D(filters=filters, kernel_size=(1, 1, 1), padding='same', activation='relu',
                                           name=name + '>depth_conv{0}'.format(index + 1))(layer)

        else:
            layer = keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                           name=name + '>conv{0}'.format(index + 1))(layer)

        if dropout:
            layer = keras.layers.SpatialDropout3D(rate=0.5)(layer)

    to_concat = layer
    layer = keras.layers.MaxPooling3D(pool_size=pool_size, strides=pool_strides, padding='same',
                                         name=name + '>pool')(layer)

    #layer = tf.keras.layers.BatchNormalization()(layer)

    return layer, to_concat


def deconv_block(n_conv, layer, to_concat, filters, kernel=3, strides=2, dropout=False, name=None, depth_wise= None, deconvolve=True):

    if deconvolve:
        layer = keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding='same',
                                                name=name + '>upconv')(layer)
    else:
        layer = keras.layers.UpSampling3D(size=strides,
                                                name=name + '>upsample')(layer)
        layer = keras.layers.Conv3D(filters=filters, kernel_size=strides, padding='same', activation='relu',
                                             name=name + '>upsamp_conv')(layer)

    if to_concat is not None:
        layer = keras.ops.concatenate([layer, to_concat], axis=-1)

    for index in range(n_conv):

        if depth_wise:
            layer = keras.layers.Conv3D(filters=filters, kernel_size=kernel, groups=filters, padding='same', activation='linear',
                                           name=name + '>vol_conv{0}'.format(index + 1))(layer)
            layer = keras.layers.Conv3D(filters=filters, kernel_size=(1, 1, 1), padding='same', activation='relu',
                                           name=name + '>depth_conv{0}'.format(index + 1))(layer)
        else:
            layer = keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                           name=name + '>conv{0}'.format(index + 1))(layer)

        if dropout:
            layer = keras.layers.SpatialDropout3D(rate=0.5)(layer)

    #layer = tf.keras.layers.BatchNormalization()(layer)

    return layer


def add_3d_coord(layer, only_z=False):
    # FIXME can we make this using a loop?
    im_shape = keras.ops.shape(layer)[1:4]
    batch_size_tensor = keras.ops.shape(layer)[0]

    # create nzyx_matrix
    xval_range = keras.ops.arange(im_shape[2])
    xval_range = keras.ops.expand_dims(xval_range, axis=0)
    xval_range = keras.ops.expand_dims(xval_range, axis=0)
    xval_range = keras.ops.expand_dims(xval_range, axis=0)
    xval_range = keras.ops.tile(xval_range, [batch_size_tensor, im_shape[0], im_shape[1], 1])

    # normalize?
    xval_range = keras.ops.cast(xval_range, 'float32')
    # add batch channel dim
    xval_range = keras.ops.expand_dims(xval_range, axis=-1)

    # create nzyx_matrix
    yval_range = keras.ops.arange(im_shape[1])
    yval_range = keras.ops.expand_dims(yval_range, axis=0)
    yval_range = keras.ops.expand_dims(yval_range, axis=0)
    yval_range = keras.ops.expand_dims(yval_range, axis=-1)
    yval_range = keras.ops.tile(yval_range, [batch_size_tensor, im_shape[0], 1, im_shape[2]])

    # normalize?
    yval_range = keras.ops.cast(yval_range, 'float32')
    # add batch channel dim
    yval_range = keras.ops.expand_dims(yval_range, axis=-1)

    # create nzyx_matrix
    zval_range = keras.ops.arange(im_shape[0])
    zval_range = keras.ops.expand_dims(zval_range, axis=0)
    zval_range = keras.ops.expand_dims(zval_range, axis=-1)
    zval_range = keras.ops.expand_dims(zval_range, axis=-1)
    zval_range = keras.ops.tile(zval_range, [batch_size_tensor, 1, im_shape[1], im_shape[2]])

    # normalize?
    zval_range = keras.ops.cast(zval_range, 'float32')
    # add batch channel dim
    zval_range = keras.ops.expand_dims(zval_range, axis=-1)

    if only_z:
        layer = keras.ops.concatenate([layer, zval_range], axis=-1)
    else:
        layer = keras.ops.concatenate([layer, zval_range, yval_range, xval_range], axis=-1)

    return layer


def tensorboard_callback(tensorboard_folder: str) -> keras.callbacks.Callback:
    return keras.callbacks.TensorBoard(
        log_dir=tensorboard_folder,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq=1000,
        embeddings_freq=0,
        embeddings_metadata=None,
    )





