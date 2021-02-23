import tensorflow as tf

from organoid_tracker.position_detection_cnn.custom_filters import local_softmax, blur_labels, disk_labels


def custom_loss(y_true, y_pred):

    non_zero_count = tf.cast(tf.math.count_nonzero(y_true), tf.float32)
    size = tf.cast(tf.size(y_true), tf.float32)
    # weight the loss by the amount of non zeroes values in label
    zero_count = tf.subtract(size, non_zero_count)

    weights = tf.where(tf.equal(y_true, 0),
                       tf.fill(tf.shape(y_pred), tf.divide(0.5 * size, zero_count)),
                       tf.fill(tf.shape(y_pred), tf.divide(0.5 * size, non_zero_count)))

    squared_difference = tf.square(y_true - y_pred)
    squared_difference = tf.multiply(weights, squared_difference)

    return tf.reduce_mean(squared_difference, axis=[1, 2, 3, 4])


def custom_loss_with_blur(y_true, y_pred):

    y_true_blur = blur_labels(y_true)

    non_zero_count = tf.cast(tf.math.count_nonzero(y_true_blur), tf.float32)
    full_size = tf.cast(tf.size(y_true_blur), tf.float32)
    zero_count = full_size - non_zero_count
    mean_labels = tf.cast(tf.reduce_mean(y_true_blur), tf.float32)

    # weight the loss by the amount of non zeroes values in label
    weights = tf.where(tf.equal(y_true_blur, 0),
                       tf.fill(tf.shape(y_true_blur), 0.5 * full_size / zero_count),
                       tf.divide(y_true_blur, mean_labels * 2))

    # Calculate weighted mean square error
    squared_difference = tf.square(y_true_blur - y_pred)
    squared_difference = tf.multiply(weights, squared_difference)
    return tf.reduce_mean(squared_difference, axis=[1, 2, 3, 4])

def peak_finding(y_pred, threshold=0.1, volume = [3, 13, 13], softmax = False):
    #local_sum = 3*13*13*tf.nn.avg_pool3d(tf.math.exp(y_pred), ksize=[3, 13, 13], strides=1, padding='SAME')
    #softmax = tf.divide(tf.math.exp(y_pred), local_sum + 1)
    if softmax:
        softmax = local_softmax(y_pred)
    else:
        softmax = y_pred

    dilation = tf.nn.max_pool3d(softmax, ksize=volume, strides=1, padding='SAME')

    peaks = tf.where(dilation == softmax, 1., 0)

    range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)

    peaks = tf.where((softmax) > threshold*range, peaks, 0)

    return peaks

def position_precision(y_true, y_pred):

    #dilation = tf.nn.max_pool3d(y_pred, ksize=[3, 13, 13], strides=1, padding='SAME')

    #peaks = tf.where(dilation == y_pred, 1., 0)
    #local_sum = 3*13*13*tf.nn.avg_pool3d(tf.math.exp(y_pred), ksize=[3, 13, 13], strides=1, padding='SAME')
    #local_sum = disk_labels(tf.math.exp(y_pred))
    #peaks = tf.where(local_sum > 1, peaks, 0)
    #softmax = tf.divide(tf.math.exp(y_pred), local_sum + 1)
    #peaks = tf.where(local_sum > 1, peaks, 0)

    peaks = peak_finding(y_pred)

    edges = get_edges(peaks)
    peaks = tf.where(edges == 0, peaks, 0)
    #y_true = tf.where(edges == 0, peaks, 0)

    y_true_blur = disk_labels(y_true)
    correct_peaks = tf.where(y_true_blur > 0, peaks, 0)

    return (tf.reduce_sum(correct_peaks)+0.01)/(tf.reduce_sum(peaks)+0.01)

def misses(y_true, y_pred):
    peaks = peak_finding(y_pred)

    peaks_blur = disk_labels(peaks)
    undetected_positions = tf.where(peaks_blur == 0, y_true, 0)

    return undetected_positions

def falses(y_true, y_pred):
    peaks = peak_finding(y_pred)
    edges = get_edges(peaks)
    peaks = tf.where(edges == 0, peaks, 0)

    positions = peak_finding(y_true, volume=[3,3,3])

    positions_blur = disk_labels(positions)
    undetected_positions = tf.where(positions_blur == 0, peaks, 0)

    return undetected_positions


def get_edges(peaks):
    edges = tf.zeros(tf.shape(peaks))
    edges = tf.pad(edges, paddings=[[0, 0], [0, 1], [1, 1], [1, 1], [0, 0]], constant_values=1)
    edges = disk_labels(edges)
    edges = edges[:, 0:-1, 1:-1, 1:-1, :]

    return edges

def position_recall(y_true, y_pred):
    peaks = peak_finding(y_pred)

    positions = peak_finding(y_true, softmax=False)

    peaks_blur = disk_labels(peaks)
    detected_positions = tf.where(peaks_blur > 0, positions, 0)

    return tf.reduce_sum(detected_positions) / tf.reduce_sum(positions)

def overcount(y_true, y_pred):
    peaks = peak_finding(y_pred)

    y_true_blur = disk_labels(y_true)
    correct_positions = tf.where(y_true_blur > 0, peaks, 0)

    positions = peak_finding(y_true, volume=[1, 3, 3], softmax=False)

    return tf.reduce_sum(correct_positions) / tf.reduce_sum(positions)


def position_loss(y_true, y_pred):

    # target image
    dilation = tf.nn.max_pool3d(y_true, ksize=[1, 3, 3], strides=1, padding='SAME')
    peaks = tf.where(dilation == y_true, 1., 0)
    y_true = tf.where(y_true > 0.1, peaks, 0)

    y_true_blur = blur_labels(y_true, kernel_size=8, sigma=2, depth=3, normalize=True)

    # weights
    weights = blur_labels(y_true, kernel_size=16, sigma=4, depth=5)
    mean_weights = tf.reduce_mean(weights)

    non_zero_count = tf.cast(tf.math.count_nonzero(y_true_blur), tf.float32)
    full_size = tf.cast(tf.size(y_true_blur), tf.float32)
    zero_count = full_size - non_zero_count

    # weight the loss by the amount of non zeroes values in label
    weights = tf.where(tf.equal(weights, 0),
                       tf.fill(tf.shape(y_true_blur), 0.5 * full_size / zero_count),
                       tf.divide(weights * 0.5, mean_weights))

    # Calculate weighted mean square error
    squared_difference = tf.square(y_true_blur - y_pred)
    squared_difference = tf.multiply(weights, squared_difference)
    return tf.reduce_mean(squared_difference, axis=[1, 2, 3, 4])


## Experimental loss functions



def new_loss(y_true, y_pred):

    max_project = tf.maximum(y_true, y_pred)
    mean = 0.5 #tf.reduce_mean(max_project)

    size = tf.cast(tf.size(y_true), tf.float32)
    low_values = tf.less(max_project, tf.fill(tf.shape(max_project), mean))
    low_count = tf.cast(tf.math.count_nonzero(low_values), tf.float32)
    high_count = size - low_count


    # weight the loss by the amount of non zeroes values in label

    weights = tf.where(low_values,
                       tf.fill(tf.shape(y_pred), tf.divide(0.5 * size, low_count)),
                       tf.fill(tf.shape(y_pred), tf.divide(0.5 * size, high_count)))

    squared_difference = tf.square(y_true - y_pred)
    squared_difference = tf.multiply(weights, squared_difference)

    return tf.reduce_mean(squared_difference, axis=[1, 2, 3, 4])


def new_loss2(y_true, y_pred):

    # weight the loss by the amount of non zeroes values in label
    y_true_blur = y_true
    alpha = tf.cast(2, tf.float32)
    summation = y_pred + y_true_blur

    #difference = tf.multiply(0.333333333, tf.math.pow(y_pred, 3)) - tf.multiply(y_pred, tf.square(y_true))

    #epsilon = 0.001

    #loss = 10000 * (tf.divide(tf.math.pow(summation, alpha+2), alpha + 2) - tf.divide(tf.multiply(2 * y_true, tf.math.pow(summation, alpha+1)), alpha + 1))
    #loss = tf.math.pow(summation, alpha+2)*(alpha+1) - tf.multiply(2 * y_true, tf.math.pow(summation, alpha+1)) * ( alpha + 2)
    #loss = tf.multiply(tf.math.pow(summation, 3), (3)*summation - (4)*2*y_pred)
    loss = tf.multiply(0.25, tf.math.pow(y_pred, 4)) \
           + tf.multiply(y_true_blur, tf.multiply(0.3333333, tf.math.pow(y_pred, 3))) \
           - tf.multiply(tf.math.pow(y_true_blur, 2), tf.multiply(0.5, tf.math.pow(y_pred, 2))) \
           - tf.multiply(tf.math.pow(y_true_blur, 3), y_pred)

    #loss = loss + epsilon * tf.square(y_pred - y_true)

    return tf.reduce_mean(loss, axis=[1, 2, 3, 4])

def KL_div_with_blur(y_true, y_pred):

    y_true_blur = blur_labels(y_true)

    non_zero_count = tf.cast(tf.math.count_nonzero(y_true_blur), tf.float32)
    size = tf.cast(tf.size(y_true_blur), tf.float32)

    y_true_blur = tf.where(tf.equal(y_true_blur, 0),
                           tf.fill(tf.shape(y_pred), tf.divide(0.01, size-non_zero_count)),
                           y_true_blur)

    y_pred = tf.where(tf.equal(y_pred, 0),
                           tf.fill(tf.shape(y_pred), tf.divide(0.01, size-non_zero_count)),
                           y_pred)

    y_true_blur = tf.divide(y_true_blur, tf.reduce_sum(y_true_blur))
    y_pred = tf.divide(y_pred, tf.reduce_sum(y_pred))

    return tf.keras.losses.kl_divergence(y_true_blur, y_pred)