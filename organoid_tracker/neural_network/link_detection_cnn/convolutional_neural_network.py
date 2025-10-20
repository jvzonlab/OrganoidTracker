from typing import Tuple

import keras
import keras.losses
import keras.metrics


def build_model(shape: Tuple, batch_size):
    shape = (shape[0], shape[1], shape[2], shape[3] + 3)

    # Input layer
    input_1 = keras.Input(shape=shape, batch_size=batch_size, name='input_1')
    input_2 = keras.Input(shape=shape, batch_size=batch_size, name='input_2')
    input_2_rev = reverse(input_2)
    input_distance = keras.Input(shape=(3,), batch_size=batch_size, name='input_distances')

    # combines inputs in channel dimension
    both_inputs = stack_target(input_1, input_2_rev)

    # define encoder, a CNN that will work the same on both input images. Encoding them to a sequence of features.
    def encoder():
        filter_sizes = [8, 16, 32]
        # filter_sizes = [16, 32, 64]

        conv_0 = conv_block(2, filters=filter_sizes[0], kernel=(1, 3, 3), pool_size=(1, 2, 2),
                            pool_strides=(1, 2, 2), name="down0")
        conv_1 = conv_block(2, filters=filter_sizes[0], name="down1")
        conv_2 = conv_block(2, filters=filter_sizes[1], name="down2")
        conv_3 = conv_block(2, filters=filter_sizes[2], name="down3")
        conv_4 = conv_block(2, filters=filter_sizes[2], name="down4", pool_size=(1, 2, 2),
                            pool_strides=(1, 2, 2))
        batch_norm = keras.layers.BatchNormalization()
        reshape = keras.layers.Reshape((filter_sizes[2] * (shape[0] // 8) * (shape[1] // 32) * (shape[2] // 32),))

        # return keras.Sequential(layers=conv_0 + conv_1 + conv_2 + conv_3 + conv_4 + [batch_norm, reshape])
        return keras.Sequential(layers=conv_0 + conv_1 + conv_2 + conv_3 + conv_4 + [reshape])

    # encoder for single images
    encoder_single = encoder()

    # encoder for combination of images
    encoder_both = encoder()

    # does a simple correlation computation on both images
    # correlation = keras.layers.Reshape((1,))(correlate(input_1, input_2))

    # concatenates encoders with correlation, distance as vector and absolute distances
    # layer = keras.layers.concatenate([encoder_single(input_1_xyz), encoder_single(input_2_rev_xyz), encoder_both(both_inputs_xyz),  input_distance, tf.abs(input_distance)])
    layer = keras.layers.concatenate(
        [encoder_single(input_1), encoder_single(input_2_rev), encoder_both(both_inputs), input_distance,
         keras.ops.abs(input_distance)])

    # dense layers
    layer = keras.layers.Dense(128, activation='relu', name='dense1')(layer)
    # layer = keras.layers.Dropout(0.5)(layer)

    layer = keras.layers.Dense(128, activation='relu', name='dense2')(layer)
    # layer = keras.layers.Dropout(0.5)(layer)

    layer = keras.layers.Dense(128, activation='relu', name='dense3')(layer)
    # layer = keras.layers.Dense(128, activation='relu', name='dense4')(layer)
    layer = keras.layers.Dense(64, activation='relu', name='dense5')(layer)
    # layer = keras.layers.Dropout(0.5)(layer)

    # output layer
    output = keras.layers.Dense(1, activation='sigmoid', name='out')(layer)

    # full model
    model = keras.Model(inputs=[input_1, input_2, input_distance], outputs=output, name="links")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00003),
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[keras.metrics.BinaryAccuracy(name='acc'),
                           keras.metrics.Recall(name='rec'),
                           keras.metrics.Precision(name='pre')])
    return model


def stack_target(tensor, tensor_target):
    return keras.ops.stack([tensor[:, :, :, :, 0], tensor_target[:, :, :, :, 0]], axis=-1)


def reverse(tensor):
    return keras.ops.flip(tensor, axis=[-1])


def conv_block(n_conv, filters, kernel=3, pool_size=2, pool_strides=2, name=None):
    layers = []
    for index in range(n_conv):
        layer = keras.layers.Conv3D(filters=filters, kernel_size=kernel, padding='same', activation='relu',
                                    name=name + '>conv{0}'.format(index + 1))
        layers.append(layer)

    layer = keras.layers.MaxPooling3D(pool_size=pool_size, strides=pool_strides, padding='same',
                                      name=name + '>pool')
    layers.append(layer)

    return layers

