"""Predictions particle positions using an already-trained convolutional neural network."""
import json
import os
from typing import Tuple

from organoid_tracker.config import ConfigFile, config_type_bool
from organoid_tracker.image_loading.channel_merging_image_loader import ChannelMergingImageLoader
from organoid_tracker.imaging import io
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.core.position_collection import PositionCollection

from organoid_tracker.link_detection_cnn.prediction_dataset import prediction_data_creator
import tensorflow as tf
import numpy as np

from organoid_tracker.link_detection_cnn.training_data_creator import create_image_with_possible_links_list

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_links")
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

_model_folder = config.get_or_prompt("checkpoint_folder", "Please paste the path here to the \"checkpoints\" folder containing the trained model.")
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

# create image_list from experiment
image_with_links_list, predicted_links_list, possible_links = create_image_with_possible_links_list(experiment)

# set relevant parameters
if not os.path.isfile(os.path.join(_model_folder, "settings.json")):
    print("Error: no settings.json found in model folder.")
    exit(1)
with open(os.path.join(_model_folder, "settings.json")) as file_handle:
    json_contents = json.load(file_handle)
    if json_contents["type"] != "links":
        print("Error: model at " + _model_folder + " is made for working with " + str(json_contents["type"]) + ", not links")
        exit(1)
    time_window = json_contents["time_window"]
    patch_shape_xyz = json_contents["patch_shape_xyz"]
    patch_shape_zyx = [patch_shape_xyz[2], patch_shape_xyz[1], patch_shape_xyz[0]]

# load models
print("Loading model...")
model = tf.keras.models.load_model(_model_folder)
if not os.path.isfile(os.path.join(_model_folder, "settings.json")):
    print("Error: no settings.json found in model folder.")
    exit(1)


print("start predicting...")
all_positions = PositionCollection()

prediction_dataset_all = prediction_data_creator(image_with_links_list, time_window, patch_shape_zyx)
predictions_all = model.predict(prediction_dataset_all)

number_of_links_done = 0

for i in range(len(image_with_links_list)):
    print("predict image {}/{}".format(i, len(image_with_links_list)))

    set_size = len(predicted_links_list[i])

    # create dataset and predict
    predictions = predictions_all[number_of_links_done : (number_of_links_done+set_size)]
    number_of_links_done = number_of_links_done + set_size

    #predictions = model.predict(prediction_dataset)
    predicted_links = predicted_links_list[i]

    for predicted_link, prediction in zip(predicted_links, predictions):
        experiment.link_data.set_link_data(predicted_link[0], predicted_link[1], data_name="link_probability", value=float(prediction))
        eps= 10 ** -10
        experiment.link_data.set_link_data(predicted_link[0], predicted_link[1], data_name="link_penalty", value=float(-np.log10(prediction+eps)+np.log10(1-prediction+eps)))

print("Saving file...")
experiment.links = possible_links
io.save_data_to_json(experiment, _output_file)
