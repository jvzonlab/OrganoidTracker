from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras



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
    layer = tf.keras.layers.Reshape((filter_sizes[2] * shape[0] // 4 * shape[1] // 16 * shape[2] // 16,))(layer)

    # layer = tf.keras.layers.Dropout(0.5)(layer)
    # layer = tf.keras.layers.Dense(128, activation='relu', name='dense0')(layer)
    # layer = tf.keras.layers.Dense(128, activation='relu', name='dense1')(layer)
    # layer = tf.keras.layers.Dense(128, activation='relu', name='dense2')(layer)
    layer = tf.keras.layers.Dense(32, activation='relu', name='dense')(layer)

    # drop-out layer, does this help?
    # layer = tf.keras.layers.Dropout(0.5)(layer)

    # sigmoid output for binary decisions
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(layer)

    # construct model
    model = keras.Model(inputs=input, outputs=output, name="YOLO_division")

    # Add loss functions and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'),
                           tf.keras.metrics.Recall(name='rec'),
                           tf.keras.metrics.Precision(name='pre')])

    return model


def conv_block(n_conv, layer, filters, kernel=3, pool_size=2, pool_strides=2, name=None):
    for index in range(n_conv):
        layer = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                       name=name + '/conv{0}'.format(index + 1))(
            layer)
        # layer = tf.keras.layers.BatchNormalization()(layer)

    layer = tf.keras.layers.MaxPooling3D(pool_size=pool_size, strides=pool_strides, padding='same',
                                         name=name + '/pool')(layer)

    return layer

# define tensorflow callback for during training
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
