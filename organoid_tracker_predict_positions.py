"""Predicts cell positions using an already-trained convolutional neural network."""
import json
import math
import os

import numpy as np
import tensorflow as tf
from skimage.feature import peak_local_max
from tifffile import tifffile

from organoid_tracker.config import ConfigFile, config_type_int
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import io
from organoid_tracker.position_detection_cnn.loss_functions import loss, position_precision, overcount, position_recall
from organoid_tracker.position_detection_cnn.peak_calling import reconstruct_volume
from organoid_tracker.position_detection_cnn.prediction_dataset import predicting_data_creator
from organoid_tracker.position_detection_cnn.split_images import corners_split, reconstruction
from organoid_tracker.position_detection_cnn.training_data_creator import create_image_list_without_positions

experiment = Experiment()

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_positions")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)

_patch_shape_z = config.get_or_default("patch_shape_z", str(30), comment="Maximum patch size to use for predictions."
                                       " Make this smaller if you run out of video card memory.", type=config_type_int)
_patch_shape_y = config.get_or_default("patch_shape_y", str(240), type=config_type_int)
_patch_shape_x = config.get_or_default("patch_shape_x", str(240), type=config_type_int)

_buffer_z = config.get_or_default("buffer_z", str(1), comment="Buffer space to use when stitching multiple patches"
                                  " together", type=config_type_int)
_buffer_y = config.get_or_default("buffer_y", str(8), type=config_type_int)
_buffer_x = config.get_or_default("buffer_x", str(8), type=config_type_int)

_model_folder = config.get_or_prompt("model_folder", "Please paste the path here to the \"trained_model\" folder containing the trained model.")
_output_file = config.get_or_default("positions_output_file", "Automatic positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")
_channels_str = config.get_or_default("images_channels", str(1), comment="Index(es) of the channels to use. Use \"3\" to use the third channel for predictions. Use \"1,3,4\" to use the sum of the first, third and fourth channel for predictions.")
_images_channels = {int(part) for part in _channels_str.split(",")}
_mid_layers = int(config.get_or_default("mid_layers", str(5), comment="Number of layers to interpolate in between"
                                        " z-planes. Used to improve peak finding."))
_peak_min_distance_px = int(config.get_or_default("peak_min_distance_px", str(9), comment="Minimum distance in pixels"
                                                  " between detected positions."))

_debug_folder = config.get_or_default("predictions_output_folder", "", comment="If you want to see the raw prediction images, paste the path to a folder here. In that folder, a prediction image will be placed for each time point.")
if len(_debug_folder) == 0:
    _debug_folder = None
config.save()
# END OF PARAMETERS


# Check if images were loaded
if not experiment.images.image_loader().has_images():
    print("No images were found. Please check the configuration file and make sure that you have stored images at"
          " the specified location.")
    exit(1)

# Edit image channels if necessary
if _images_channels != {1}:
    # Replace the first channel
    old_channels = experiment.images.get_channels()
    new_channels = [old_channels[index - 1] for index in _images_channels]
    channel_merging_image_loader = ChannelSummingImageLoader(experiment.images.image_loader(), [new_channels])
    experiment.images.image_loader(channel_merging_image_loader)

# create image_list from experiment
image_list = create_image_list_without_positions(experiment)

# set relevant parameters
patch_shape = [_patch_shape_z, _patch_shape_y, _patch_shape_x]
buffer = np.array([[_buffer_z, _buffer_z], [_buffer_y, _buffer_y], [_buffer_x, _buffer_x]])

# find the maximum image size to use as the basis for splitting
max_image_shape = np.zeros(3, dtype=int)
for i in range(len(image_list)):
    image_shape = np.asarray(image_list[i].get_image_size_zyx())
    max_image_shape = np.maximum(max_image_shape, image_shape)

# define how to split input
corners = corners_split(max_image_shape, patch_shape)

# due to memory constraints only ~10 images can be processed at a given time (depending on patch shape)
set_size = 2

# set relevant parameters
if not os.path.isfile(os.path.join(_model_folder, "settings.json")):
    print("Error: no settings.json found in model folder.")
    exit(1)
with open(os.path.join(_model_folder, "settings.json")) as file_handle:
    json_contents = json.load(file_handle)
    if json_contents["type"] != "positions":
        print("Error: model is made for working with " + str(json_contents["type"]) + ", not positions")
        exit(1)
    time_window = json_contents["time_window"]

# load models
print("Loading model...")
model = tf.keras.models.load_model(_model_folder, custom_objects={"loss": loss,
                                                                  "position_precision": position_precision,
                                                                  "position_recall": position_recall,
                                                                  "overcount": overcount})
if _debug_folder is not None:
    os.makedirs(_debug_folder, exist_ok=True)

print("Starting predictions...")
all_positions = PositionCollection()

# Create iterator over complete dataset
prediction_dataset_all = predicting_data_creator(image_list, time_window, corners,
                                             patch_shape, buffer, max_image_shape, batch_size = set_size * len(corners))
prediction_dataset_all_iter = iter(prediction_dataset_all)

# Go over all images
image_set_count = int(math.ceil(len(image_list) / set_size))

for image_set_index in range(image_set_count):

    # pick part of the image_list
    if (set_size * image_set_index) < len(image_list):
        image_list_subset = image_list[image_set_index * set_size: (image_set_index + 1) * set_size]
    else:
        image_list_subset = image_list[image_set_index * set_size:]

    print(f"Predicting set of images {image_set_index + 1}/{image_set_count}")

    # set current set size
    current_set_size = min(len(image_list)-image_set_index*set_size, set_size)

    # take relevant part of the tf.Dataset
    prediction_dataset = prediction_dataset_all_iter.get_next()
    prediction_dataset = tf.data.Dataset.from_tensor_slices(prediction_dataset).batch(1)

    # create prediction mask for peak_finding
    prediction_mask_shape = 2*tf.floor(np.sqrt(_peak_min_distance_px ** 2 / 3)) + 1
    prediction_mask = np.ones([int(prediction_mask_shape),]*3)

    # make predictions
    predictions = model.predict(prediction_dataset)

    # split set in batches of patches belonging to single figure
    predictions = np.split(predictions, current_set_size)

    for image, prediction_batch in zip(image_list_subset, predictions):
        # register image information
        time_point = image.time_point
        image_offset = image.get_image_offset()
        image_shape = list(image.get_image_size_zyx())

        # reconstruct image from patches
        prediction = reconstruction(prediction_batch, corners, buffer, image_shape, patch_shape)
        # remove channel dimension
        prediction = np.squeeze(prediction, axis=-1)

        if _debug_folder is not None:
        #if _debug_folder is None:
            image_name = "image_" + str(time_point.time_point_number())
            tifffile.imsave(os.path.join(_debug_folder, '{}.tif'.format(image_name)), prediction)

        del prediction_batch

        # peak detection
        print(f"Detecting peaks at time point {time_point.time_point_number()}...")
        im, z_divisor = reconstruct_volume(prediction, _mid_layers)  # interpolate between layers for peak detection

        # Comparison between image_max and im to find the coordinates of local maxima
        #im = erosion(im, np.ones((7,7,7)))
        coordinates = peak_local_max(im, min_distance=_peak_min_distance_px, threshold_abs=0.1,  exclude_border=False) #, footprint=prediction_mask)

        for coordinate in coordinates:
            pos = Position(coordinate[2], coordinate[1], coordinate[0] / z_divisor - 1,
                           time_point=time_point) + image_offset
            all_positions.add(pos)

experiment.positions.add_positions(all_positions)

print("Saving file...")
io.save_data_to_json(experiment, _output_file)


