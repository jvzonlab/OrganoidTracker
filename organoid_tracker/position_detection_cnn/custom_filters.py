from typing import Tuple, Optional

import tensorflow as tf


def local_softmax(y_pred: tf.Tensor, radius: Optional[float] = None, volume_zyx_px: Tuple[int, int, int] = (3, 13, 13), exponentiate: bool = False, blur: bool = True):
    """Calculates the softmax for the given preditions. Softmax shows soft peaks on the
     locations of the maxima. Either radius or volume_zyx_px is used."""
    if exponentiate:
        y_pred = tf.math.exp(y_pred)

    if radius is None:
        local_sum = volume_zyx_px[0] * volume_zyx_px[1] * volume_zyx_px[2] * tf.nn.avg_pool3d(y_pred, ksize=volume_zyx_px, strides=1, padding='SAME')
    else:
        local_sum = disk_labels(y_pred, radius_um=radius)

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
    """Blurs the given labels with a Gaussian kernel."""
    blur = _gaussian_kernel(kernel_size = kernel_size, sigma=sigma, depth=depth, n_channels=1, dtype=label.dtype, normalize=normalize)

    label_blur = tf.nn.conv3d(label, blur, [1, 1, 1, 1, 1], 'SAME')

    return label_blur


def _disk(radius_um: float, resolution_zyx: Tuple[float, float, float], n_channels: int = 1):
    z_range = tf.floor(radius_um / resolution_zyx[0])
    z = tf.range(-z_range, z_range+1) * resolution_zyx[0]

    y_range = tf.floor(radius_um / resolution_zyx[1])
    y = tf.range(-y_range, y_range+1) * resolution_zyx[1]

    x_range = tf.floor(radius_um / resolution_zyx[2])
    x = tf.range(-x_range, x_range+1) * resolution_zyx[2]

    Z, Y, X = tf.meshgrid(z, x, y, indexing='ij')

    distance = tf.square(Z) + tf.square(Y) + tf.square(X)

    disk = tf.where(distance < tf.square(radius_um), 1., 0.)

    disk = tf.expand_dims(disk, axis=-1)
    return tf.expand_dims(tf.tile(disk, (1, 1, 1, n_channels)), axis=-1)


def disk_labels(label: tf.Tensor, radius_um=5.0):
    """Converts labels into disks."""
    disk = _disk(radius_um=radius_um, resolution_zyx=(2, 0.4, 0.4))

    label_disk = tf.nn.conv3d(label, disk, [1, 1, 1, 1, 1], 'SAME')

    return label_disk


def peak_finding_radius(y_pred: tf.Tensor, radius: float = 4.5, tolerance: float = 0.01, threshold: float = 0.01):

    n = tf.cast(tf.reduce_sum(_disk(radius_um=radius, resolution_zyx=(2, 0.4, 0.4))), tf.float32)
    range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
    alpha = tf.math.log(n) / (range * tolerance)

    y_pred_exp = tf.math.exp(alpha * y_pred)
    local_logexpsum = 1/alpha * tf.math.log(disk_labels(y_pred_exp, radius_um=radius))

    peaks = tf.where(local_logexpsum - tolerance * range <= y_pred, y_pred, 0)

    peaks = tf.where(peaks > threshold * range + tf.reduce_min(y_pred), 1, 0)

    return peaks


