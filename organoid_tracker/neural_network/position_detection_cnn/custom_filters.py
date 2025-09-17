from typing import Tuple

import keras

from organoid_tracker.neural_network import Tensor


def local_softmax(y_pred: Tensor, volume_zyx_px: Tuple[int, int, int] = (3, 13, 13), exponentiate: bool = False, blur: bool = True):
    """Calculates the softmax for the given preditions. Softmax shows soft peaks on the
     locations of the maxima. Either radius or volume_zyx_px is used."""
    if exponentiate:
        y_pred = keras.ops.where(y_pred > 10, 10., y_pred)
        y_pred = keras.ops.exp(y_pred)

    local_sum = volume_zyx_px[0] * volume_zyx_px[1] * volume_zyx_px[2] * keras.ops.average_pool(y_pred, pool_size=volume_zyx_px, strides=1, padding='same')

    softmax = keras.ops.divide(y_pred, local_sum + 1)

    if blur:
        softmax = blur_labels(softmax, kernel_size=8, depth=3, sigma=2.)
    else:
        scale = keras.ops.sum(_gaussian_kernel(8, 2., 3, 1, dtype="float32"))
        softmax = scale * softmax

    return softmax


def _gaussian_kernel(kernel_size: int, sigma: float, depth: int, n_channels: int, dtype: str, normalize: bool = True):
    """Creates kernel in the XY plane"""
    x = keras.ops.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = keras.ops.exp(-(keras.ops.power(x, 2) / (2 * keras.ops.power(keras.ops.cast(sigma, dtype), 2))))
    g_kernel = keras.ops.tensordot(g, g, axes=0)

    if depth == 3:
        g_kernel = keras.ops.stack([0.25 * g_kernel, 0.5 * g_kernel, 0.25 * g_kernel])
    elif depth == 5:
        g_kernel = keras.ops.stack([0.1 * g_kernel, 0.2 * g_kernel, 0.4 * g_kernel, 0.2 * g_kernel, 0.1 * g_kernel])
    else:
        g_kernel = keras.ops.expand_dims(g_kernel, axis=0)

    # scale so maximum is at 1.
    if normalize:
        g_kernel = g_kernel / keras.ops.max(g_kernel)
    else:
        g_kernel = g_kernel / keras.ops.sum(g_kernel)

    # add channel dimension and later batch dimension
    g_kernel = keras.ops.expand_dims(g_kernel, axis=-1)
    return keras.ops.expand_dims(keras.ops.tile(g_kernel, (1, 1, 1, n_channels)), axis=-1)


def blur_labels(label: Tensor, kernel_size: int = 8, sigma: float = 2.0, depth: int = 3, n_channels: int = 1, normalize: bool = True):
    """Blurs the given labels (single pixels) with a Gaussian kernel."""
    blur = _gaussian_kernel(kernel_size = kernel_size, sigma=sigma, depth=depth, n_channels=n_channels, dtype=label.dtype, normalize=normalize)

    label_blur = keras.ops.conv(label, blur, [1, 1, 1], 'same')

    return label_blur

def _disk(range_zyx: Tuple[float, float, float] = (2.5, 11., 11.), n_channels: int = 1):
    range_int = keras.ops.floor(range_zyx)

    z = keras.ops.arange(-range_int[0], range_int[0] + 1) / range_zyx[0]

    y = keras.ops.arange(-range_int[1], range_int[1] + 1) / range_zyx[1]

    x = keras.ops.arange(-range_int[2], range_int[2] + 1) / range_zyx[2]

    Z, Y, X = keras.ops.meshgrid(z, x, y, indexing='ij')

    distance = keras.ops.square(Z) + keras.ops.square(Y) + keras.ops.square(X)

    disk = keras.ops.where(distance < 1, 1., 0.)

    disk = keras.ops.expand_dims(disk, axis=-1)
    return keras.ops.expand_dims(keras.ops.tile(disk, (1, 1, 1, n_channels)), axis=-1)


#def disk_labels(label, range_zyx: Tuple[float, float, float] = (5., 11., 11.)):
def disk_labels(label, range_zyx: Tuple[float, float, float] = (2.5, 11., 11.)):
    """Changes the labels (single pixels) into disks."""
    disk = _disk(range_zyx= range_zyx)

    label_disk = keras.ops.conv(label, disk, [1, 1, 1], 'same')

    return label_disk

def get_edges(peaks, range_zyx: Tuple[float, float, float] = (2.5, 11., 11.), remove_top=True):
    edges = keras.ops.zeros(keras.ops.shape(peaks))
    if remove_top:
        edges = keras.ops.pad(edges, [[0, 0], [0, 1], [1, 1], [1, 1], [0, 0]], constant_values=1)
        edges = disk_labels(edges, range_zyx=range_zyx)
        edges = edges[:, 0:-1, 1:-1, 1:-1, :]
    else:
        edges = keras.ops.pad(edges, [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], constant_values=1)
        edges = disk_labels(edges, range_zyx=range_zyx)
        edges = edges[:, :, 1:-1, 1:-1, :]

    return edges

def get_edges_with_bottom(peaks, range_zyx: Tuple[float, float, float] = (2.5, 11., 11.)):
    edges = keras.ops.zeros(keras.ops.shape(peaks))
    edges = keras.ops.pad(edges, paddings=[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], constant_values=1)
    edges = disk_labels(edges, range_zyx=range_zyx)
    edges = edges[:, 1:-1, 1:-1, 1:-1, :]

    return edges

def _disk_res(radius=5.0, resolution =[2, 0.4, 0.4], n_channels=1):
    z_range = keras.ops.floor(radius/resolution[0])
    z = keras.ops.arange(-z_range, z_range+1)*resolution[0]

    y_range = keras.ops.floor(radius/resolution[1])
    y = keras.ops.arange(-y_range, y_range+1)*resolution[1]

    x_range = keras.ops.floor(radius/resolution[2])
    x = keras.ops.arange(-x_range, x_range+1)*resolution[2]

    Z, Y, X = keras.ops.meshgrid(z, x, y, indexing='ij')

    distance = keras.ops.square(Z) + keras.ops.square(Y) + keras.ops.square(X)

    disk = keras.ops.where(distance < keras.ops.square(radius), 1., 0.)

    disk = keras.ops.expand_dims(disk, axis=-1)
    return keras.ops.expand_dims(keras.ops.tile(disk, (1, 1, 1, n_channels)), axis=-1)


def _distance(range = [3., 13., 13.],  n_channels=1, squared=True):

    range_int = keras.ops.round(range)

    z = keras.ops.arange(-range_int[0], range_int[0] + 1) / range[0]

    y = keras.ops.arange(-range_int[1], range_int[1] + 1) / range[1]

    x = keras.ops.arange(-range_int[2], range_int[2] + 1) / range[2]

    Z, Y, X = keras.ops.meshgrid(z, x, y, indexing='ij')

    distance = keras.ops.square(Z) + keras.ops.square(Y) + keras.ops.square(X)

    distance = keras.ops.where(distance > 1, 1., distance)

    if not squared:
        distance = keras.ops.sqrt(distance)

    distance = keras.ops.expand_dims(distance, axis=-1)

    return keras.ops.expand_dims(keras.ops.tile(distance, (1, 1, 1, n_channels)), axis=-1)


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
    distances_min = keras.ops.conv(y_true, keras.ops.exp(distance*k)-1, [1, 1, 1], 'same') + 1
    # d* = log(sum(exp(d*k)-1)))/k
    distances_min = keras.ops.log(distances_min) / k
    distances_min = keras.ops.where(keras.ops.isinf(distances_min), 1., distances_min)

    # sum of the inverted distance to center points
    distance_sum = keras.ops.conv(y_true, distance, [1, 1, 1], 'same')

    if adaptive:
        # create distance map
        distances = 1 - 2 * distances_min + distance_sum
        distances = keras.ops.where(distances > 1, 1., distances)

        #s_squared = 0.25 # c Elegans
        s_squared = 0.125
        #s_squared = 0.125/2
        distances = keras.ops.exp(-distances ** 2 / (2*s_squared))

        # normalize
        distances = (distances - keras.ops.exp(- 1./ (2*s_squared))) / (1.- keras.ops.exp(- 1./ (2*s_squared)))
    else:
        new_range = [range[0]/2, range[1]/2, range[2]/2]
        #new_range = [range[0]*0.75, range[1]*0.75, range[2]*0.75]
        distance = 1 - _distance(range=new_range, squared=False)

        distances = 1- keras.ops.conv(y_true, distance, [1, 1, 1], 'same')
        s_squared = 0.25
        distances = keras.ops.exp(-distances ** 2 / (2 * s_squared))
        distances = (distances - keras.ops.exp(- 1. / (2 * s_squared))) / (1. - keras.ops.exp(- 1. / (2 * s_squared)))

    # define weights based on inverted distance sum
    #s_squared_weight = 0.125/2 fly
    s_squared_weight = 0.125
    #s_squared_weight = 0.25
    #s_squared_weight = 0.5 # fly embryo

    #new_range = [range[0]*1.5, range[1]*1.5, range[2]*1.5] fly
    #distance = 1 - _distance(range=new_range, squared=False)
    #distance_sum = keras.ops.nn.conv3d(y_true, distance, [1, 1, 1, 'same')

    weights = 1-keras.ops.minimum(distance_sum, 1.0)
    weights = keras.ops.exp(-weights ** 2 / (2*s_squared_weight))
    weights = (weights - keras.ops.exp(- 1 / (2*s_squared_weight))) / (1 - keras.ops.exp(- 1 / (2*s_squared_weight)))

    # define background as zero
    weights = keras.ops.where(weights < 0.05, 0., weights)

    # give background equal weight to foreground
    non_zero_count = keras.ops.cast(keras.ops.count_nonzero(weights), "float32")

    full_size = keras.ops.cast(keras.ops.size(weights), "float32")
    zero_count = full_size - non_zero_count

    background_weight = 0.5
    #background_weight = 0.0 #fly
    # background_weight = 0.95 #worm
    # weight the loss by the amount of non zeroes values in label
    weights = keras.ops.where(keras.ops.equal(weights, 0),
                        background_weight * full_size / zero_count,
                       keras.ops.divide(weights * (1-background_weight), keras.ops.mean(weights)))

    weights = keras.ops.where(keras.ops.less(weights,  background_weight * full_size / zero_count),
                       background_weight * full_size / zero_count,
                       weights)

    # set weight edges to zero
    #edges = get_edges_with_bottom(weights, range_zyx=range_edges)
    edges = get_edges(weights, range_zyx=range_edges, remove_top=False)
    weights = keras.ops.where(edges > 0, 0., weights)

    weights = keras.ops.divide(weights, keras.ops.mean(weights))

    return distances, weights
