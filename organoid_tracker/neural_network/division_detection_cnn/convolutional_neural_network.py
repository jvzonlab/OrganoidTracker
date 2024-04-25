from typing import List, Tuple

import keras
import keras.metrics
import keras.losses


def build_model(shape: Tuple, batch_size):
    # Input layer
    input = keras.Input(shape=shape, batch_size=batch_size)

    # convolutions
    filter_sizes = [32, 32, 64]
    # filter_sizes = [16, 32, 32]

    layer = conv_block(2, input, filters=filter_sizes[0], kernel=(2, 2, 2), pool_size=(1, 2, 2),
                       pool_strides=(1, 2, 2), name="down0")
    layer = conv_block(2, layer, filters=filter_sizes[0], kernel=(2, 2, 2), pool_size=(1, 2, 2),
                       pool_strides=(1, 2, 2), name="down1")
    layer = conv_block(2, layer, filters=filter_sizes[1], name="down2")
    layer = conv_block(2, layer, filters=filter_sizes[2], name="down3")

    # reshape 4x4 image to vector
    layer = keras.layers.Reshape((filter_sizes[2] * shape[0] // 4 * shape[1] // 16 * shape[2] // 16,))(layer)

    # layer = keras.layers.Dropout(0.5)(layer)
    # layer = keras.layers.Dense(128, activation='relu', name='dense0')(layer)
    # layer = keras.layers.Dense(128, activation='relu', name='dense1')(layer)
    # layer = keras.layers.Dense(128, activation='relu', name='dense2')(layer)
    layer = keras.layers.Dense(32, activation='relu', name='dense')(layer)

    # drop-out layer, does this help?
    # layer = keras.layers.Dropout(0.5)(layer)

    # sigmoid output for binary decisions
    output = keras.layers.Dense(1, activation='sigmoid', name='out')(layer)

    # construct model
    model = keras.Model(inputs=input, outputs=output, name="YOLO_division")

    # Add loss functions and metrics
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[keras.metrics.BinaryAccuracy(name='acc'),
                           keras.metrics.Recall(name='rec'),
                           keras.metrics.Precision(name='pre')])

    return model


def conv_block(n_conv, layer, filters, kernel=3, pool_size=2, pool_strides=2, name=None):
    for index in range(n_conv):
        layer = keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                       name=name + '/conv{0}'.format(index + 1))(
            layer)
        # layer = keras.layers.BatchNormalization()(layer)

    layer = keras.layers.MaxPooling3D(pool_size=pool_size, strides=pool_strides, padding='same',
                                         name=name + '/pool')(layer)

    return layer
