"""" Loads images and positions into tensors and can write these tensors into TFR files"""
from typing import List
from functools import partial

from organoid_tracker.neural_network.division_detection_cnn.training_data_creator import _ImageWithDivisions
from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import ImageWithPositions


# Loads images with position and division information
def load_images_with_divisions(image_with_positions: _ImageWithDivisions, time_window=(0, 0), create_labels=False):
    image = image_with_positions.load_image_time_stack(time_window)

    # change positions into label image (pixels in image)
    label = image_with_positions.xyz_positions
    label = label[:, [2,1,0]]

    # extract division info
    dividing = image_with_positions.dividing

    return image, label, dividing

# tensorflow wrapper to laod image + division data
def tf_load_images_with_divisions(i, image_with_positions_list: List[ImageWithPositions], time_window=[0, 0], create_labels=False):
    if create_labels:
        image, label, dividing = tf.py_function(
            partial(load_images_with_divisions, image_with_positions_list=image_with_positions_list,
                time_window=time_window, create_labels=create_labels), [i],
            (tf.float32, tf.float32, tf.bool))
    else:
        image, label, dividing = tf.py_function(
            partial(load_images_with_divisions, image_with_positions_list=image_with_positions_list,
                time_window=time_window, create_labels=create_labels), [i],
            (tf.float32, tf.int32, tf.bool))

    return image, label, dividing


def load_images_with_positions(i, image_with_positions_list: List[_ImageWithDivisions], time_window=[0, 0], create_labels=False):

    image_with_positions = image_with_positions_list[i]

    image = image_with_positions.load_image_time_stack(time_window)
    # change positions in to label image (pixels in image)
    if create_labels:
        label = image_with_positions.create_labels(image.shape[0:3])
    # reorder the position information to zyx
    else:

        label = image_with_positions.xyz_positions
        label = label[:, [2,1,0]]

    return image, label

# tensorflow wrapper to laod image + positions
def tf_load_images_with_positions(i, image_with_positions_list: List[ImageWithPositions], time_window=[0, 0], create_labels=False):
    if create_labels:
        image, label = tf.py_function(
            partial(load_images_with_positions, image_with_positions_list=image_with_positions_list,
                time_window=time_window, create_labels=create_labels), [i],
            (tf.float32, tf.float32))
    else:
        image, label = tf.py_function(
            partial(load_images_with_positions, image_with_positions_list=image_with_positions_list,
                time_window=time_window, create_labels=create_labels), [i],
            (tf.float32, tf.int32))

    return image, label


