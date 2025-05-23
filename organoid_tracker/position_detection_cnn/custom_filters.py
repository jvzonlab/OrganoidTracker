from typing import Tuple

import tensorflow as tf


def local_softmax(y_pred: tf.Tensor, volume_zyx_px: Tuple[int, int, int] = (3, 13, 13), exponentiate: bool = False, blur: bool = True):
    """Calculates the softmax for the given preditions. Softmax shows soft peaks on the
     locations of the maxima. Either radius or volume_zyx_px is used."""
    if exponentiate:
        y_pred = tf.where(y_pred > 10, 10., y_pred)
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
    else:
        g_kernel = g_kernel / tf.reduce_sum(g_kernel)

    # add channel dimension and later batch dimension
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, 1, n_channels)), axis=-1)


def blur_labels(label: tf.Tensor, kernel_size: int = 8, sigma: float = 2.0, depth: int = 3, n_channels: int = 1, normalize: bool = True):
    """Blurs the given labels (single pixels) with a Gaussian kernel."""
    blur = _gaussian_kernel(kernel_size = kernel_size, sigma=sigma, depth=depth, n_channels=n_channels, dtype=label.dtype, normalize=normalize)

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


#def disk_labels(label, range_zyx: Tuple[float, float, float] = (5., 11., 11.)):
def disk_labels(label, range_zyx: Tuple[float, float, float] = (2.5, 11., 11.)):
    """Changes the labels (single pixels) into disks."""
    disk = _disk(range_zyx= range_zyx)

    label_disk = tf.nn.conv3d(label, disk, [1, 1, 1, 1, 1], 'SAME')

    return label_disk

def get_edges(peaks, range_zyx: Tuple[float, float, float] = (2.5, 11., 11.), remove_top=True):
    edges = tf.zeros(tf.shape(peaks))
    if remove_top:
        edges = tf.pad(edges, paddings=[[0, 0], [0, 1], [1, 1], [1, 1], [0, 0]], constant_values=1)
        edges = disk_labels(edges, range_zyx=range_zyx)
        edges = edges[:, 0:-1, 1:-1, 1:-1, :]
    else:
        edges = tf.pad(edges, paddings=[[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], constant_values=1)
        edges = disk_labels(edges, range_zyx=range_zyx)
        edges = edges[:, :, 1:-1, 1:-1, :]

    return edges

def get_edges_with_bottom(peaks, range_zyx: Tuple[float, float, float] = (2.5, 11., 11.)):
    edges = tf.zeros(tf.shape(peaks))
    edges = tf.pad(edges, paddings=[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], constant_values=1)
    edges = disk_labels(edges, range_zyx=range_zyx)
    edges = edges[:, 1:-1, 1:-1, 1:-1, :]

    return edges

def _disk_res(radius=5.0, resolution =[2, 0.4, 0.4], n_channels=1):
    z_range = tf.floor(radius/resolution[0])
    z = tf.range(-z_range, z_range+1)*resolution[0]

    y_range = tf.floor(radius/resolution[1])
    y = tf.range(-y_range, y_range+1)*resolution[1]

    x_range = tf.floor(radius/resolution[2])
    x = tf.range(-x_range, x_range+1)*resolution[2]

    Z, Y, X = tf.meshgrid(z, x, y, indexing='ij')

    distance = tf.square(Z) + tf.square(Y) + tf.square(X)

    disk = tf.where(distance < tf.square(radius), 1., 0.)

    disk = tf.expand_dims(disk, axis=-1)
    return tf.expand_dims(tf.tile(disk, (1, 1, 1, n_channels)), axis=-1)


def peak_finding_radius(y_pred, radius = 4.5, tolerance = 0.01, threshold = 0.01):

    n=tf.cast(tf.reduce_sum(_disk(radius=radius)), tf.float32)
    range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
    alpha = tf.math.log(n) / (range * tolerance)

    y_pred_exp = tf.math.exp(alpha * y_pred)
    local_logexpsum = 1/alpha * tf.math.log(disk_labels(y_pred_exp, radius=radius))

    peaks = tf.where(local_logexpsum - tolerance * range <= y_pred, y_pred, 0)

    peaks = tf.where(peaks > threshold * range + tf.reduce_min(y_pred), 1, 0)

    return peaks


def _distance(range = [3., 13., 13.],  n_channels=1, squared=True):

    range_int = tf.round(range)

    z = tf.range(-range_int[0], range_int[0] + 1) / range[0]

    y = tf.range(-range_int[1], range_int[1] + 1) / range[1]

    x = tf.range(-range_int[2], range_int[2] + 1) / range[2]

    Z, Y, X = tf.meshgrid(z, x, y, indexing='ij')

    distance = tf.square(Z) + tf.square(Y) + tf.square(X)

    distance = tf.where(distance > 1, 1., distance)

    if not squared:
        distance = tf.sqrt(distance)

    distance = tf.expand_dims(distance, axis=-1)

    return tf.expand_dims(tf.tile(distance, (1, 1, 1, n_channels)), axis=-1)


def distance_map(y_true, range=(3., 16., 16.), range_edges = (3., 11. , 11.), adaptive = True):
#def distance_map(y_true, range=(3., 16., 16.), range_edges = (3., 11. , 11.), adaptive = False): #unadapted
#def distance_map(y_true, range=(4., 16., 16.), range_edges=(3., 11., 11.), adaptive=True): # fly embryo
#def distance_map(y_true, range=(3., 12., 12.), range_edges=(3., 8., 8.), adaptive=True): # lower resolution

    # exponent used to take pseudo_maximum
    k = 20.

    # invert distances
    distance = 1 - _distance(range = range, squared=False)

    # take pseudo maximum of the inverted distance to center points
    # d* = sum(exp(d*k)-1))
    distances_min = tf.nn.conv3d(y_true, tf.exp(distance*k)-1, [1, 1, 1, 1, 1], 'SAME') + 1
    # d* = log(sum(exp(d*k)-1)))/k
    distances_min = tf.math.log(distances_min) / k
    distances_min = tf.where(tf.math.is_inf(distances_min), 1., distances_min)

    # sum of the inverted distance to center points
    distance_sum = tf.nn.conv3d(y_true, distance, [1, 1, 1, 1, 1], 'SAME')

    if adaptive:
        # create distance map
        distances = 1 - 2 * distances_min + distance_sum
        distances = tf.where(distances > 1, 1., distances)

        #s_squared = 0.25 # c Elegans
        s_squared = 0.2 # Human intestinal organoids
        #s_squared = 0.125
        #s_squared = 0.125/2
        distances = tf.exp(-distances ** 2 / (2*s_squared))

        # normalize
        distances = (distances - tf.exp(- 1./ (2*s_squared))) / (1.- tf.exp(- 1./ (2*s_squared)))
    else:
        new_range = [range[0]/2, range[1]/2, range[2]/2]
        #new_range = [range[0]*0.75, range[1]*0.75, range[2]*0.75]
        distance = 1 - _distance(range=new_range, squared=False)

        distances = 1- tf.nn.conv3d(y_true, distance, [1, 1, 1, 1, 1], 'SAME')
        s_squared = 0.25
        distances = tf.exp(-distances ** 2 / (2 * s_squared))
        distances = (distances - tf.exp(- 1. / (2 * s_squared))) / (1. - tf.exp(- 1. / (2 * s_squared)))

    # define weights based on inverted distance sum
    #s_squared_weight = 0.125/2 fly
    s_squared_weight = 0.125
    #s_squared_weight = 0.25
    #s_squared_weight = 0.5 # fly embryo

    #new_range = [range[0]*1.5, range[1]*1.5, range[2]*1.5] fly
    #distance = 1 - _distance(range=new_range, squared=False)
    #distance_sum = tf.nn.conv3d(y_true, distance, [1, 1, 1, 1, 1], 'SAME')

    weights = 1-tf.math.minimum(distance_sum, 1.0)
    weights = tf.exp(-weights ** 2 / (2*s_squared_weight))
    weights = (weights - tf.exp(- 1 / (2*s_squared_weight))) / (1 - tf.exp(- 1 / (2*s_squared_weight)))

    # define background as zero
    weights = tf.where(weights < 0.05, 0., weights)

    # give background equal weight to foreground
    non_zero_count = tf.cast(tf.math.count_nonzero(weights), tf.float32)

    full_size = tf.cast(tf.size(weights), tf.float32)
    zero_count = full_size - non_zero_count

    background_weight = 0.5
    #background_weight = 0.0 #fly
    # background_weight = 0.95 #worm
    # weight the loss by the amount of non zeroes values in label
    weights = tf.where(tf.equal(weights, 0),
                        background_weight * full_size / zero_count,
                       tf.divide(weights * (1-background_weight), tf.reduce_mean(weights)))

    weights = tf.where(tf.less(weights,  background_weight * full_size / zero_count),
                       background_weight * full_size / zero_count,
                       weights)

    # set weight edges to zero
    #edges = get_edges_with_bottom(weights, range_zyx=range_edges)
    edges = get_edges(weights, range_zyx=range_edges, remove_top=False)
    weights = tf.where(edges > 0, 0., weights)

    weights = tf.divide(weights, tf.reduce_mean(weights))

    return distances, weights

