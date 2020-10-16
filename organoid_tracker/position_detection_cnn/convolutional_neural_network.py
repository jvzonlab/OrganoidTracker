from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras

from organoid_tracker.position_detection_cnn.loss_functions import custom_loss_with_blur, new_loss2


def build_model(shape: Tuple, batch_size):
    # Input layer
    input = keras.Input(shape=shape, batch_size=batch_size)

    # Add coordinates
    layer = add_3d_coord(input)

    # convolutions
    to_concat = []

    filter_sizes = [3, 16, 64, 128, 256]
    #filter_sizes = [3, 16, 32, 64, 128]
    #filter_sizes = [3, 16, 32, 32, 32]
    layer, to_concat_layer = conv_block(2, layer, filters=filter_sizes[1], kernel=(1, 3, 3), pool_size=(1, 2, 2),
                                        pool_strides=(1, 2, 2), name="down1")
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(2, layer, filters=filter_sizes[2], name="down2")
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(2, layer, filters=filter_sizes[3], name="down3")
    to_concat.append(to_concat_layer)
    layer, to_concat_layer = conv_block(2, layer, filters=filter_sizes[4], name="down4")
    to_concat.append(to_concat_layer)

    layer = deconv_block(2, layer, to_concat.pop(), filters=filter_sizes[4], name="up1")
    layer = deconv_block(2, layer, to_concat.pop(), filters=filter_sizes[3], name="up2")
    layer = deconv_block(2, layer, to_concat.pop(), filters=filter_sizes[2], name="up3")
    layer = deconv_block(2, layer, to_concat.pop(), filters=filter_sizes[1], kernel=(1, 3, 3), strides=(1, 2, 2), name="up4")

    # apply final batch_normalization
    layer = tf.keras.layers.BatchNormalization()(layer)

    output = tf.keras.layers.Conv3D(filters=1, kernel_size=3, padding="same", activation='relu', name='out_conv')(layer)

    model = keras.Model(inputs=input, outputs=output, name="YOLO")

    #model.compile(optimizer='Adam', loss=keras.losses.mean_squared_error,metrics=custom_loss)
    model.compile(optimizer='Adam', loss=custom_loss_with_blur, metrics=new_loss2)
    #model.compile(optimizer='Adam', loss=new_loss2, metrics=custom_loss)

    return model


def conv_block(n_conv, layer, filters, kernel=3, pool_size=2, pool_strides=2, name=None):
    for index in range(n_conv):
        layer = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                       name=name + '/conv{0}'.format(index + 1))(
            layer)  # To test : is coordconv needed in all layers or just first?
        #layer = tf.keras.layers.BatchNormalization()(layer)

    to_concat = layer
    layer = tf.keras.layers.MaxPooling3D(pool_size=pool_size, strides=pool_strides, padding='same',
                                         name=name + '/pool')(layer)

    return layer, to_concat


def deconv_block(n_conv, layer, to_concat, filters, kernel=3, strides=2, name=None):
    layer = tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding='same',
                                            name=name + '/upconv')(layer)
    #layer = tf.keras.layers.BatchNormalization()(layer)

    for index in range(n_conv):
        layer = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                       name=name + '/conv{0}'.format(index + 1))(layer)
        #layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.concat([layer, to_concat], axis=-1)

    return layer


def add_3d_coord(layer):
    # FIXME can we make this using a loop?
    im_shape = tf.shape(layer)[1:4]
    batch_size_tensor = tf.shape(layer)[0]

    # create nzyx_matrix
    xval_range = tf.range(im_shape[2])
    xval_range = tf.expand_dims(xval_range, axis=0)
    xval_range = tf.expand_dims(xval_range, axis=0)
    xval_range = tf.expand_dims(xval_range, axis=0)
    xval_range = tf.tile(xval_range, [batch_size_tensor, im_shape[0], im_shape[1], 1])

    # normalize?
    xval_range = tf.cast(xval_range, 'float32')
    # add batch channel dim
    xval_range = tf.expand_dims(xval_range, axis=-1)

    # create nzyx_matrix
    yval_range = tf.range(im_shape[1])
    yval_range = tf.expand_dims(yval_range, axis=0)
    yval_range = tf.expand_dims(yval_range, axis=0)
    yval_range = tf.expand_dims(yval_range, axis=-1)
    yval_range = tf.tile(yval_range, [batch_size_tensor, im_shape[0], 1, im_shape[2]])

    # normalize?
    yval_range = tf.cast(yval_range, 'float32')
    # add batch channel dim
    yval_range = tf.expand_dims(yval_range, axis=-1)

    # create nzyx_matrix
    zval_range = tf.range(im_shape[0])
    zval_range = tf.expand_dims(zval_range, axis=0)
    zval_range = tf.expand_dims(zval_range, axis=-1)
    zval_range = tf.expand_dims(zval_range, axis=-1)
    zval_range = tf.tile(zval_range, [batch_size_tensor, 1, im_shape[1], im_shape[2]])

    # normalize?
    zval_range = tf.cast(zval_range, 'float32')
    # add batch channel dim
    zval_range = tf.expand_dims(zval_range, axis=-1)

    layer = tf.concat([layer, xval_range, yval_range, zval_range], axis=-1)
    return layer


tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=0,
    write_graph=False,
    write_images=False,
    update_freq=1000,
    profile_batch=(100, 105),
    embeddings_freq=0,
    embeddings_metadata=None,
)


