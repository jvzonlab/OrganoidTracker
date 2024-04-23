from typing import List

import keras
import torch.nn.functional

from organoid_tracker.neural_network.position_detection_cnn.custom_filters import blur_labels, disk_labels, distance_map, \
    get_edges
from organoid_tracker.neural_network import Tensor


def peak_finding(y_pred: Tensor, threshold: float = 0.1, volume: List[int] = (3, 13, 13)):
    """Finds the local peaks in the given predictions. Operates by doing a dilation,
    and then checking where the actual value reaches the dilation."""
    dilation = torch.nn.functional.max_pool3d(y_pred, volume, stride=1, padding=tuple(v // 2 for v in volume))
    # The following line should also work, but for some reason the padding is miscalculated as [6, 6, 1]
    # instead of [1, 6, 6] by Keras. Bug in Keras?
    # dilation = keras.ops.max_pool(y_pred, volume, strides=1, padding='same')

    peaks = keras.ops.where(dilation == y_pred, 1., 0)
    range = keras.ops.max(y_pred) - keras.ops.min(y_pred)
    peaks = keras.ops.where(y_pred > threshold * range, peaks, 0)
    return peaks


def position_precision(y_true: Tensor, y_pred: Tensor):
    """"""
    # y_true = interpolate_z(y_true)
    # edges = get_edges_with_bottom(y_true, range_zyx=(2.5, 16., 16.))
    edges = get_edges(y_true, range_zyx=(2.5, 16., 16.))

    y_pred = keras.ops.where(edges == 0, y_pred, 0)

    peaks = peak_finding(y_pred)

    y_true_blur = disk_labels(y_true)
    correct_peaks = keras.ops.where(y_true_blur > 0, peaks, 0)

    return (keras.ops.sum(correct_peaks) + 0.001) / (keras.ops.sum(peaks) + 0.001)


def position_recall(y_true, y_pred):
    # y_true = interpolate_z(y_true)

    # edges = get_edges_with_bottom(y_true, range_zyx=(2.5, 16., 16.))
    edges = get_edges(y_true, range_zyx=(2.5, 16., 16.))

    y_true = keras.ops.where(edges == 0, y_true, 0)
    y_pred = keras.ops.where(edges == 0, y_pred, 0)

    peaks = peak_finding(y_pred)
    positions = peak_finding(y_true)

    # edges = get_edges_with_bottom(peaks)
    # peaks = tf.where(edges == 0, peaks, 0)
    # positions = tf.where(edges == 0, positions, 0)

    peaks_blur = disk_labels(peaks)
    detected_positions = keras.ops.where(peaks_blur > 0, positions, 0)

    return keras.ops.sum(detected_positions) / keras.ops.sum(positions)


def overcount(y_true, y_pred):
    # y_true = interpolate_z(y_true)
    edges = get_edges(y_true, range_zyx=(2.5, 16., 16.))
    y_pred = keras.ops.where(edges == 0, y_pred, 0)
    peaks = peak_finding(y_pred)

    y_true_blur = disk_labels(y_true)
    correct_positions = keras.ops.where(y_true_blur > 0, peaks, 0)

    positions = peak_finding(y_true)
    peaks_blur = disk_labels(peaks)
    detected_positions = keras.ops.where(peaks_blur > 0, positions, 0)

    return (keras.ops.sum(correct_positions) + 0.001) / (keras.ops.sum(detected_positions) + 0.001)


def loss(y_true, y_pred):
    # find cell centers on target after distortions (rotation/scaling)
    dilation = torch.nn.functional.max_pool3d(y_pred, (1, 3, 3), stride=1, padding=(0, 1, 1))
    # The following line should also work, but for some reason the padding is miscalculated as [1, 1, 0]
    # instead of [0, 1, 1] by Keras. Bug in Keras?
    # dilation = keras.ops.max_pool(y_true, [1, 3, 3], strides=1, padding='same')
    peaks = keras.ops.where(dilation == y_true, 1., 0)
    y_true = keras.ops.where(y_true > 0.1, peaks, 0)

    # create target image
    dist_map, weights = distance_map(y_true)

    dist_map = blur_labels(dist_map, sigma=1.5, kernel_size=4, depth=1, normalize=False)
    # dist_map= blur_labels(dist_map, sigma=3, kernel_size=7, depth=3, normalize=False)

    # Calculate weighted mean square error
    squared_difference = keras.ops.square(dist_map - y_pred)
    squared_difference = keras.ops.multiply(weights, squared_difference)

    return keras.ops.mean(squared_difference, axis=[1, 2, 3, 4])
