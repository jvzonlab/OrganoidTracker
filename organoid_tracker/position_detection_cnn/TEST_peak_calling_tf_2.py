import tifffile as tiff
import numpy as np
import tensorflow as tf

from organoid_tracker.position_detection_cnn.loss_functions import blur_labels, peak_finding, position_recall, \
    position_precision, overcount, misses, falses, get_edges

input = tiff.imread('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_input7.tiff')

input = tf.expand_dims(input, axis=-1)
input = tf.expand_dims(input, axis=0)

image = tiff.imread('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_prediction7.tiff')
image = tf.convert_to_tensor(image)
label = tiff.imread('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_labels7.tiff')
print(tf.shape(image))
image = tf.expand_dims(image, axis=-1)
image = tf.expand_dims(image, axis=0)

local_sum = 3 * 13 * 13 * tf.nn.avg_pool3d(tf.math.exp(image), ksize=[3, 13, 13], strides=1, padding='SAME')
softmax = tf.divide(tf.math.exp(image), local_sum + 1)
softmax = blur_labels(softmax)

label = tf.expand_dims(label, axis=-1)
label = tf.expand_dims(label, axis=0)

peaks = peak_finding(image)

print(position_recall(label, image))
print(position_precision(label, image))
print(overcount(label, image))
missed = misses(label, image)
falsed = falses(label, image)
edges = get_edges(label)

missed = blur_labels(missed)
falsed = blur_labels(falsed)
peaks = blur_labels(peaks)
label = blur_labels(label)

combo = tf.stack([peaks, label, input], axis=2)
combo = tf.squeeze(combo, axis=-1)



print(tf.shape(combo))

tiff.imsave('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_peaks7.tiff', peaks.numpy())
tiff.imsave('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_softmax7.tiff', softmax.numpy())
tiff.imwrite('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_combo7.tiff', combo.numpy(),
            imagej=True, metadata={'axes': 'TZCYX'})
tiff.imwrite('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_misses7.tiff', missed.numpy())
tiff.imwrite('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_falses7.tiff', falsed.numpy())
tiff.imwrite('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_edges7.tiff', edges.numpy())


#tiff.imsave('X:/Neural_network/Xuan_Rutger_Guizela_position/t=0_t=1/output/examples/example_dilation7.tiff',
            #tf.math.exp(dilation).numpy())
