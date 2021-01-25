import tensorflow as tf

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


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    # creates kernel in the XY plane
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_kernel = tf.tensordot(g, g, axes=0)

    g_kernel = tf.stack([0.1 * g_kernel, 0.25 * g_kernel, 0.5 * g_kernel, 0.25 * g_kernel, 0.1 * g_kernel])
    #g_kernel = tf.stack([0.25 * g_kernel, 0.5 * g_kernel, 0.25 * g_kernel])
    # scale so maximum is at 1.
    g_kernel = g_kernel / tf.reduce_max(g_kernel)

    # add channel dimension and later batch dimension
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, 1, n_channels)), axis=-1)


def blur_labels(label):
    blur = _gaussian_kernel(kernel_size=12, sigma=3, n_channels=1, dtype=label.dtype)

    label_blur = tf.nn.conv3d(label, blur, [1, 1, 1, 1, 1], 'SAME')

    return label_blur


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
