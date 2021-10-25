from typing import List

import tensorflow as tf

from organoid_tracker.position_detection_cnn.custom_filters import blur_labels, disk_labels, distance_map, \
    _gaussian_kernel


def peak_finding(y_pred: tf.Tensor, threshold: float = 0.1, volume: List[int] =[3, 13, 13]):
    """Finds the local peaks in the given predictions. Operates by doing a dilation,
    and then checking where the actual value reaches the dilation."""
    dilation = tf.nn.max_pool3d(y_pred, ksize=volume, strides=1, padding='SAME')
    peaks = tf.where(dilation == y_pred, 1., 0)
    range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
    peaks = tf.where((y_pred) > threshold*range, peaks, 0)
    return peaks


def position_precision(y_true: tf.Tensor, y_pred: tf.Tensor):
    """"""
    peaks = peak_finding(y_pred)

    edges = get_edges(peaks)
    peaks = tf.where(edges == 0, peaks, 0)

    y_true_blur = disk_labels(y_true)
    correct_peaks = tf.where(y_true_blur > 0, peaks, 0)

    return (tf.reduce_sum(correct_peaks)+0.01)/(tf.reduce_sum(peaks)+0.01)

def get_edges(peaks):
    edges = tf.zeros(tf.shape(peaks))
    edges = tf.pad(edges, paddings=[[0, 0], [0, 1], [1, 1], [1, 1], [0, 0]], constant_values=1)
    edges = disk_labels(edges)
    edges = edges[:, 0:-1, 1:-1, 1:-1, :]

    return edges

def position_recall(y_true, y_pred):
    peaks = peak_finding(y_pred)

    positions = peak_finding(y_true)

    peaks_blur = disk_labels(peaks)
    detected_positions = tf.where(peaks_blur > 0, positions, 0)

    return tf.reduce_sum(detected_positions) / tf.reduce_sum(positions)


def overcount(y_true, y_pred):
    peaks = peak_finding(y_pred)

    y_true_blur = disk_labels(y_true)
    correct_positions = tf.where(y_true_blur > 0, peaks, 0)

    positions = peak_finding(y_true, volume=[1, 3, 3], threshold=0.1)

    return tf.reduce_sum(correct_positions) / tf.reduce_sum(positions)

def loss(y_true, y_pred):
    # find cell centers on target after distortions (rotation/scaling)
    dilation = tf.nn.max_pool3d(y_true, ksize=[1, 3, 3], strides=1, padding='SAME')
    peaks = tf.where(dilation == y_true, 1., 0)
    y_true = tf.where(y_true > 0.1, peaks, 0)

    # create target image
    dist_map, weights = distance_map(y_true)

    # Calculate weighted mean square error
    squared_difference = tf.square(dist_map - y_pred)

    squared_difference = tf.multiply(weights, squared_difference)

    return tf.reduce_mean(squared_difference, axis=[1, 2, 3, 4])

