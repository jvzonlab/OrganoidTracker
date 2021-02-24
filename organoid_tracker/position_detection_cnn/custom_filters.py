from typing import Tuple

import tensorflow as tf


def local_softmax(y_pred: tf.Tensor, volume_zyx_px: Tuple[int, int, int] = (3, 13, 13), exponentiate: bool = False, blur: bool = True):
    """Calculates the softmax for the given preditions. Softmax shows soft peaks on the
     locations of the maxima. Either radius or volume_zyx_px is used."""
    if exponentiate:
        y_pred = tf.math.exp(y_pred)

    local_sum = volume_zyx_px[0] * volume_zyx_px[1] * volume_zyx_px[2] * tf.nn.avg_pool3d(y_pred, ksize=volume_zyx_px, strides=1, padding='SAME')

    softmax = tf.divide(y_pred, local_sum + 1)

    if blur:
        softmax = blur_labels(softmax, kernel_size=8, depth=3, sigma=2.)
    else:
        scale = tf.reduce_sum(_gaussian_kernel(8, 2., 3, 1, dtype=tf.float32))
        softmax = scale * softmax

    return softmax


def _gaussian_kernel(kernel_size: int, sigma: float, depth: int, n_channels: int, dtype: tf.DType, normalize: bool = True):
    """Creates kernel in the XY plane"""
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_kernel = tf.tensordot(g, g, axes=0)

    if depth == 3:
        g_kernel = tf.stack([0.25 * g_kernel, 0.5 * g_kernel, 0.25 * g_kernel])
    elif depth == 5:
        g_kernel = tf.stack([0.1 * g_kernel, 0.2 * g_kernel, 0.4 * g_kernel, 0.2 * g_kernel, 0.1 * g_kernel])
    else:
        g_kernel = tf.expand_dims(g_kernel, axis=0)

    # scale so maximum is at 1.
    if normalize:
        g_kernel = g_kernel / tf.reduce_max(g_kernel)

    # add channel dimension and later batch dimension
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, 1, n_channels)), axis=-1)


def blur_labels(label: tf.Tensor, kernel_size: int = 8, sigma: float = 2.0, depth: int = 3, normalize: bool = True):
    """Blurs the given labels (single pixels) with a Gaussian kernel."""
    blur = _gaussian_kernel(kernel_size = kernel_size, sigma=sigma, depth=depth, n_channels=1, dtype=label.dtype, normalize=normalize)

    label_blur = tf.nn.conv3d(label, blur, [1, 1, 1, 1, 1], 'SAME')

    return label_blur


def _disk(range_zyx: Tuple[float, float, float] = (2.5, 11., 11.), n_channels: int = 1):
    range_int = tf.floor(range_zyx)

    z = tf.range(-range_int[0], range_int[0] + 1) / range_zyx[0]

    y = tf.range(-range_int[1], range_int[1] + 1) / range_zyx[1]

    x = tf.range(-range_int[2], range_int[2] + 1) / range_zyx[2]

    Z, Y, X = tf.meshgrid(z, x, y, indexing='ij')

    distance = tf.square(Z) + tf.square(Y) + tf.square(X)

    disk = tf.where(distance < 1, 1., 0.)

    disk = tf.expand_dims(disk, axis=-1)
    return tf.expand_dims(tf.tile(disk, (1, 1, 1, n_channels)), axis=-1)


def disk_labels(label, range_zyx: Tuple[float, float, float] = (2.5, 11., 11.)):
    """Changes the labels (single pixels) into disks."""
    disk = _disk(range_zyx= range_zyx)

    label_disk = tf.nn.conv3d(label, disk, [1, 1, 1, 1, 1], 'SAME')

    return label_disk

