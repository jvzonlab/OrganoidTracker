"""Predictions particle positions using an already-trained convolutional neural network."""
import _keras_environment
_keras_environment.activate()

import json
import os

import keras.saving
import numpy as np

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import TimePoint
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import io, list_io
from organoid_tracker.linking.nearby_position_finder import find_closest_n_positions
from organoid_tracker.neural_network.division_detection_cnn.prediction_dataset import prediction_data_creator
from organoid_tracker.neural_network.division_detection_cnn.training_data_creator import \
    create_image_with_positions_list

# PARAMETERS

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_divisions")
_dataset_file = config.get_or_prompt("dataset_file", "Please paste the path here to the dataset file."
                                     " You can generate such a file from OrganoidTracker using File -> Tabs -> "
                                     " all tabs.", store_in_defaults=True)

_patch_shape_z = int(config.get_or_default("patch_shape_z", str(30), store_in_defaults=True))
_patch_shape_y = int(config.get_or_default("patch_shape_y", str(240), store_in_defaults=True))
_patch_shape_x = int(config.get_or_default("patch_shape_x", str(240), store_in_defaults=True))

_model_folder = config.get_or_prompt("model_folder", "Please paste the path here to the \"trained_model\" folder containing the trained model.")
_output_folder = config.get_or_default("divisions_output_folder", "Division predictions", comment="Output folder for the divisions, can be viewed using the visualizer program.")
_channels_str = config.get_or_default("images_channels", str(1), comment="Index(es) of the channels to use. Use \"3\" to use the third channel for predictions. Use \"1,3,4\" to use the sum of the first, third and fourth channel for predictions.")
_images_channels = {int(part) for part in _channels_str.split(",")}

_min_distance_dividing = float(config.get_or_default("minimum_distance_between_dividing_cell (in micron)", str(4.5)))

config.save()
# END OF PARAMETERS

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
model = keras.saving.load_model(os.path.join(_model_folder, "model.keras"))
if not os.path.isfile(os.path.join(_model_folder, "settings.json")):
    print("Error: no settings.json found in model folder.")
    exit(1)

# create output folder
os.makedirs(_output_folder, exist_ok=True)

experiments_to_save = list()
for experiment_index, experiment in enumerate(list_io.load_experiment_list_file(_dataset_file)):
    print(f"Working on experiment {experiment_index + 1}: {experiment.name}")
    # Check if images were loaded
    if not experiment.images.image_loader().has_images():
        print(f"No images were found for experiment \"{experiment.name}\". Please check the configuration file and make"
              f" sure that you have stored images at the specified location.")
        exit(1)
    experiment.images.resolution()  # Check if resolution is set

    # Edit image channels if necessary
    if _images_channels != {1}:
        # Replace the first channel
        old_channels = experiment.images.get_channels()
        new_channels = [old_channels[index - 1] for index in _images_channels]
        channel_merging_image_loader = ChannelSummingImageLoader(experiment.images.image_loader(), [new_channels])
        experiment.images.image_loader(channel_merging_image_loader)

    # create image_list from experiment, with positions_list
    image_with_positions_list, positions_list = create_image_with_positions_list(experiment)

    # create PositionCollection to store positions with division scores
    all_positions = PositionCollection()

    prediction_dataset_all = prediction_data_creator(image_with_positions_list, time_window, patch_shape)
    predictions_all = model.predict(prediction_dataset_all)

    number_of_positions_done = 0

    # predict for every time point
    for i in range(len(image_with_positions_list)):
        image_with_positions = image_with_positions_list[i]
        set_size = len(positions_list[i])

        # Extract relevant data
        predictions = predictions_all[number_of_positions_done : (number_of_positions_done+set_size)]
        number_of_positions_done = number_of_positions_done + set_size

        # get positions
        positions = positions_list[i]

        for position, prediction in zip(positions, predictions):
            eps = 10 ** -10
            likelihood = intercept + scaling * float(np.log10(prediction + eps) - np.log10(1 - prediction + eps))
            scaled_prediction = (10**likelihood) / (1 + 10**likelihood)

            # add division prediction to the data
            experiment.position_data.set_position_data(position, data_name="division_probability", value=float(scaled_prediction))
            # add division penalty (log-likelihood) to the data
            eps = 10 ** -10
            experiment.position_data.set_position_data(position, data_name="division_penalty", value=float(-likelihood))

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
                    detection_range = 1.5*_min_distance_dividing
                else:
                    detection_range = _min_distance_dividing

                # if the division is oversegmented replace by one position in the middle
                if position.distance_um(neighbor, resolution=experiment.images.resolution()) < detection_range:

                    if (position not in to_remove) and (neighbor not in to_remove):
                        add_position = Position(x=(position.x + neighbor.x) // 2,
                                                y=(position.y + neighbor.y) // 2,
                                                z=(position.z + neighbor.z) // 2,
                                                time_point=position.time_point())
                        to_add.append(add_position)

                        experiment.position_data.set_position_data(add_position, 'division_probability',
                                                                   max(experiment.position_data.get_position_data(
                                                                       position, data_name="division_probability"),
                                                                       experiment.position_data.get_position_data(
                                                                           neighbor, data_name="division_probability")))
                        experiment.position_data.set_position_data(add_position, 'division_penalty',
                                                                   min(experiment.position_data.get_position_data(
                                                                       position, data_name="division_penalty"),
                                                                       experiment.position_data.get_position_data(
                                                                           neighbor, data_name="division_penalty")))
                        # print(position)

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
                    to_remove = to_remove + [neighbor]

    # adapt positions
    experiment.remove_positions(to_remove)
    experiment.positions.add_positions(PositionCollection(to_add))

    print(f'Division oversegmentations removed: {len(to_remove)}')

    print("Saving file...")
    output_file = os.path.join(_output_folder, f"{experiment_index + 1}. {experiment.name.get_save_name()}."
                               + io.FILE_EXTENSION)
    io.save_data_to_json(experiment, output_file)
    experiments_to_save.append(experiment)


list_io.save_experiment_list_file(experiments_to_save,
                                  os.path.join(_output_folder, "_All" + list_io.FILES_LIST_EXTENSION))
print("Done!")