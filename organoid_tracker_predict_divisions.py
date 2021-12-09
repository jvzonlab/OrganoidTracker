"""Predictions particle positions using an already-trained convolutional neural network."""
import os
from typing import Tuple

from organoid_tracker.config import ConfigFile, config_type_bool
from organoid_tracker.image_loading.channel_merging_image_loader import ChannelMergingImageLoader
from organoid_tracker.imaging import io
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.core.position_collection import PositionCollection

from organoid_tracker.division_detection_cnn.prediction_dataset import prediction_data_creator
from organoid_tracker.division_detection_cnn.training_data_creator import create_image_with_positions_list

import tensorflow as tf
import numpy as np

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_divisions")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_positions_file = config.get_or_default("positions_file",
                                            "Where are the cell postions saved?",
                                            comment="What are the detected positions for those images?")

_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))

experiment = io.load_data_file(_positions_file, _min_time_point, _max_time_point)
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)

_patch_shape_z = int(config.get_or_default("patch_shape_z", str(30), store_in_defaults=True))
_patch_shape_y = int(config.get_or_default("patch_shape_y", str(240), store_in_defaults=True))
_patch_shape_x = int(config.get_or_default("patch_shape_x", str(240), store_in_defaults=True))

_time_window_before = int(config.get_or_default("time_window_before", str(-1), store_in_defaults=True))
_time_window_after = int(config.get_or_default("time_window_after", str(1), store_in_defaults=True))

_checkpoint_folder = config.get_or_prompt("checkpoint_folder", "Please paste the path here to the \"checkpoints\" folder containing the trained model.")
_output_file = config.get_or_default("positions_output_file", "Automatic positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")
_channels_str = config.get_or_default("images_channels", str(1), comment="Index(es) of the channels to use. Use \"3\" to use the third channel for predictions. Use \"1,3,4\" to use the sum of the first, third and fourth channel for predictions.")
_images_channels = {int(part) for part in _channels_str.split(",")}

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
    channel_merging_image_loader = ChannelMergingImageLoader(experiment.images.image_loader(), [new_channels])
    experiment.images.image_loader(channel_merging_image_loader)

# create image_list from experiment, with positions_list
image_with_positions_list, positions_list = create_image_with_positions_list(experiment)

# set relevant parameters
time_window = [_time_window_before, _time_window_after]
patch_shape = [_patch_shape_z, _patch_shape_y, _patch_shape_x]

# load model
print("load model...")
model = tf.keras.models.load_model(_checkpoint_folder) #, custom_objects={"new_loss2": new_loss2, "custom_loss_with_blur": custom_loss_with_blur})

print("start predicting...")

# create PositionCollection to store positions with division scores
all_positions = PositionCollection()

# predict for every time point
for i in range(len(image_with_positions_list)):
    image_with_positions = image_with_positions_list[i]
    print(image_with_positions.xyz_positions.shape)
    print("predict image {}/{}".format(i, len(image_with_positions_list)))

    # register image information
    time_point = image_with_positions._time_point
    image_offset = image_with_positions._images.offsets.of_time_point(time_point)

    # create tf.dataset and predict
    prediction_dataset = prediction_data_creator([image_with_positions], time_window, patch_shape)
    predictions = model.predict(prediction_dataset)

    # get positions
    positions = positions_list[i]

    for positions, prediction in zip(positions, predictions):
        # add division prediction to the data
        experiment.position_data.set_position_data(positions, data_name="division_probability", value=float(prediction))
        # add division penalty (log-likelihood) to the data
        eps = 10 ** -10
        experiment.position_data.set_position_data(positions, data_name="division_penalty", value=float(-np.log10(prediction+eps)+np.log10(1-prediction+eps)))


print("Saving file...")
io.save_data_to_json(experiment, _output_file)


