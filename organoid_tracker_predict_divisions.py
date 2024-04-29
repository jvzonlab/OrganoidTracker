"""Predictions particle positions using an already-trained convolutional neural network."""
import json
import os
from typing import Tuple

from organoid_tracker.config import ConfigFile, config_type_bool
from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import io
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.core.position_collection import PositionCollection

from organoid_tracker.division_detection_cnn.prediction_dataset import prediction_data_creator
from organoid_tracker.division_detection_cnn.training_data_creator import create_image_with_positions_list

import tensorflow as tf
import numpy as np

# PARAMETERS
from organoid_tracker.linking.nearby_position_finder import find_closest_n_positions

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

_pixel_size_x_um = config.get_or_default("pixel_size_x_um", "",
                                         comment="Resolution of the images. Only used if the image files and"
                                                 " tracking files don't provide a resolution.", store_in_defaults=True)
_pixel_size_y_um = config.get_or_default("pixel_size_y_um", "", store_in_defaults=True)
_pixel_size_z_um = config.get_or_default("pixel_size_z_um", "", store_in_defaults=True)
_time_point_duration_m = config.get_or_default("time_point_duration_m", "", store_in_defaults=True)
if _pixel_size_x_um and _pixel_size_y_um and _pixel_size_z_um and _time_point_duration_m:
    fallback_resolution = ImageResolution(float(_pixel_size_x_um), float(_pixel_size_y_um), float(_pixel_size_z_um),
                                          float(_time_point_duration_m))
else:
    fallback_resolution = None

experiment = io.load_data_file(_positions_file, _min_time_point, _max_time_point)
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)

# Try to fix missing resolution (this allows running all scripts in sequence)
if experiment.images.resolution(allow_incomplete=True).is_incomplete():
    if fallback_resolution is None:
        print("Please provide a resolution in the tracking data file, or in the configuration file.")
        exit(1)
    experiment.images.set_resolution(fallback_resolution)

_patch_shape_z = int(config.get_or_default("patch_shape_z", str(30), store_in_defaults=True))
_patch_shape_y = int(config.get_or_default("patch_shape_y", str(240), store_in_defaults=True))
_patch_shape_x = int(config.get_or_default("patch_shape_x", str(240), store_in_defaults=True))

_model_folder = config.get_or_prompt("checkpoint_folder", "Please paste the path here to the \"checkpoints\" folder containing the trained model.")
_output_file = config.get_or_default("positions_output_file", "Automatic positions.aut", comment="Output file for the positions, can be viewed using the visualizer program.")
_channels_str = config.get_or_default("images_channels", str(1), comment="Index(es) of the channels to use. Use \"3\" to use the third channel for predictions. Use \"1,3,4\" to use the sum of the first, third and fourth channel for predictions.")
_images_channels = {int(part) for part in _channels_str.split(",")}

_min_distance_dividing = float(config.get_or_default("minimum_distance_between_dividing_cell (in micron)", str(4.5)))

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

# create image_list from experiment, with positions_list
image_with_positions_list, positions_list = create_image_with_positions_list(experiment)

# set relevant parameters
with open(os.path.join(_model_folder, "settings.json")) as file_handle:
    json_contents = json.load(file_handle)
    if json_contents["type"] != "divisions":
        print("Error: model at " + _model_folder + " is made for working with " + str(json_contents["type"]) + ", not divisions")
        exit(1)
    time_window = json_contents["time_window"]
    patch_shape = json_contents["patch_shape_zyx"]

    scaling = json_contents["platt_scaling"] if "platt_scaling" in json_contents else 1
    intercept = json_contents["platt_intercept"] if "platt_intercept" in json_contents else 0
    intercept = np.log10(np.exp(intercept))

# load model
print("Loading model...")
model = tf.keras.models.load_model(_model_folder)
if not os.path.isfile(os.path.join(_model_folder, "settings.json")):
    print("Error: no settings.json found in model folder.")
    exit(1)

print("start predicting...")

# create PositionCollection to store positions with division scores
all_positions = PositionCollection()

prediction_dataset_all = prediction_data_creator(image_with_positions_list, time_window, patch_shape)
predictions_all = model.predict(prediction_dataset_all)

number_of_positions_done = 0

# predict for every time point
for i in range(len(image_with_positions_list)):
    image_with_positions = image_with_positions_list[i]

    print("predict image {}/{}".format(i, len(image_with_positions_list)))

    set_size = len(positions_list[i])

    # Extract relevant data
    predictions = predictions_all[number_of_positions_done : (number_of_positions_done+set_size)]
    number_of_positions_done = number_of_positions_done + set_size

    # get positions
    positions = positions_list[i]

    for positions, prediction in zip(positions, predictions):
        eps = 10 ** -10
        likelihood = intercept + scaling * float(np.log10(prediction + eps) - np.log10(1 - prediction + eps))
        scaled_prediction = (10**likelihood) / (1 + 10**likelihood)

        # add division prediction to the data
        experiment.position_data.set_position_data(positions, data_name="division_probability", value=float(scaled_prediction))
        # add division penalty (log-likelihood) to the data
        eps = 10 ** -10
        experiment.position_data.set_position_data(positions, data_name="division_penalty", value=float(-likelihood))

# Remove oversegmentation for dividing cells by setting a minimal distance for dividing cells
to_remove = []
to_add = []

for position in experiment.positions:

    # check if cell is dividing
    if experiment.position_data.get_position_data(position, data_name="division_probability") > 0.5:

        # find 6 closest neighbors
        neighbors = find_closest_n_positions(experiment.positions.of_time_point(position.time_point()), around = position, max_amount=6, resolution=experiment.images.resolution())
        for neighbor in neighbors:

            # check if neighbor is also dividing and if the neighbor is within the range
            if experiment.position_data.get_position_data(neighbor, data_name="division_probability") > 0.5:
                range = 1.5*_min_distance_dividing
            else:
                range = _min_distance_dividing

            # if the division is oversegmented rpelace by one position in the middle
            if (position.distance_um(neighbor, resolution=experiment.images.resolution()) < range):
                add_position = Position(x=(position.x + neighbor.x)//2,
                                    y=(position.y + neighbor.y)//2,
                                    z=(position.z + neighbor.z)//2,
                                    time_point=position.time_point())
                to_add.append(add_position)

                print('remove:')
                print(position)
                print('remove:')
                print(neighbor)
                print('add:')
                print(add_position)

                experiment.position_data.set_position_data(add_position, 'division_probability',
                                                           max(experiment.position_data.get_position_data(position,
                                                                                                          data_name="division_probability"),
                                                               experiment.position_data.get_position_data(neighbor,
                                                                                                          data_name="division_probability")))
                experiment.position_data.set_position_data(add_position, 'division_penalty',
                                                           min(experiment.position_data.get_position_data(position,
                                                                                                          data_name="division_penalty"),
                                                               experiment.position_data.get_position_data(neighbor,
                                                                                                          data_name="division_penalty")))
                #print(position)
                to_remove = to_remove + [position, neighbor]

        # find closest neighbors at previous timepoint
        prev_time_point = TimePoint(position.time_point().time_point_number()-1)
        neighbors = list(find_closest_n_positions(experiment.positions.of_time_point(prev_time_point), around = position, max_amount=6, resolution=experiment.images.resolution()))

        if len(neighbors)>0:
            closest_neighbor = list(find_closest_n_positions(experiment.positions.of_time_point(prev_time_point), around=position, max_amount=1,
                                             resolution=experiment.images.resolution()))[0]
        else:
            closest_neighbor = None

        # check for distances between cells in the previous frame
        for neighbor in neighbors:
            distance = position.distance_um(neighbor, resolution=experiment.images.resolution())

            # remove oversegmentation in previous frame
            if (distance < _min_distance_dividing) and (neighbor != closest_neighbor):
                print(neighbor)
                to_remove = to_remove + [neighbor]

# adapt positions
experiment.remove_positions(to_remove)
experiment.positions.add_positions(PositionCollection(to_add))

print('division oversegmentations removed:')
print(len(to_remove))

print("Saving file...")
io.save_data_to_json(experiment, _output_file)


