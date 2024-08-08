"""Predictions particle positions using an already-trained convolutional neural network."""
import _keras_environment
_keras_environment.activate()

import json
import os

import keras.saving
import numpy as np

from organoid_tracker.config import ConfigFile
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import io, list_io
from organoid_tracker.neural_network.link_detection_cnn.prediction_dataset import prediction_data_creator
from organoid_tracker.neural_network.link_detection_cnn.training_data_creator import \
    create_image_with_possible_links_list

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_links")
_dataset_file = config.get_or_prompt("dataset_file", "Please paste the path here to the dataset file."
                                     " You can generate such a file from OrganoidTracker using File -> Tabs -> "
                                     " all tabs.", store_in_defaults=True)

_model_folder = config.get_or_prompt("model_folder", "Please paste the path here to the \"trained_model\" folder containing the trained model.")
_output_folder = config.get_or_default("predictions_output_folder", "Link predictions", comment="Output folder for the links, can be viewed using the visualizer program.")
_channels_str = config.get_or_default("images_channels", str(1), comment="Index(es) of the channels to use. Use \"3\" to use the third channel for predictions. Use \"1,3,4\" to use the sum of the first, third and fourth channel for predictions.")
_images_channels = {int(part) for part in _channels_str.split(",")}

config.save()
# END OF PARAMETERS


# set relevant parameters
if not os.path.isfile(os.path.join(_model_folder, "settings.json")):
    print("Error: no settings.json found in model folder.")
    exit(1)
with open(os.path.join(_model_folder, "settings.json")) as file_handle:
    json_contents = json.load(file_handle)
    if json_contents["type"] != "links":
        print("Error: model at " + _model_folder + " is made for working with " + str(
            json_contents["type"]) + ", not links")
        exit(1)
    time_window = json_contents["time_window"]
    if "patch_shape_xyz" in json_contents:
        patch_shape_xyz = json_contents["patch_shape_xyz"]
        patch_shape_zyx = [patch_shape_xyz[2], patch_shape_xyz[1], patch_shape_xyz[0]]
    else:
        patch_shape_zyx = json_contents["patch_shape_zyx"]  # Seems like some versions of OrganoidTracker use this
    scaling = json_contents["platt_scaling"] if "platt_scaling" in json_contents else 1
    intercept = json_contents["platt_intercept"] if "platt_intercept" in json_contents else 0
    intercept = np.log10(np.exp(intercept))

# load models
print("Loading model...")
model = keras.saving.load_model(os.path.join(_model_folder, "model.keras"))
if not os.path.isfile(os.path.join(_model_folder, "settings.json")):
    print("Error: no settings.json found in model folder.")
    exit(1)

# Create output folder
os.makedirs(_output_folder, exist_ok=True)

# Loop through experiments
experiments_to_save = list()
for experiment_index, experiment in enumerate(list_io.load_experiment_list_file(_dataset_file)):
    print(f"Working on experiment {experiment_index + 1}: {experiment.name}")

    # Check if images were loaded
    if not experiment.images.image_loader().has_images():
        print("No images were found. Please check the configuration file and make sure that you have stored images at"
              " the specified location.")
        exit(1)
    experiment.images.resolution()  # Check for resolution

    # Edit image channels if necessary
    if _images_channels != {1}:
        # Replace the first channel
        old_channels = experiment.images.get_channels()
        new_channels = [old_channels[index - 1] for index in _images_channels]
        channel_merging_image_loader = ChannelSummingImageLoader(experiment.images.image_loader(), [new_channels])
        experiment.images.image_loader(channel_merging_image_loader)

    # create image_list from experiment
    print("Building link list...")
    image_with_links_list, predicted_links_list, possible_links = create_image_with_possible_links_list(experiment)

    print("Start predicting...")
    all_positions = PositionCollection()

    prediction_dataset_all = prediction_data_creator(image_with_links_list, time_window, patch_shape_zyx)
    predictions_all = model.predict(prediction_dataset_all)

    number_of_links_done = 0

    print("Storing predictions in experiment...")
    for i in range(len(image_with_links_list)):
        set_size = len(predicted_links_list[i])

        # create dataset and predict
        predictions = predictions_all[number_of_links_done : (number_of_links_done+set_size)]
        number_of_links_done = number_of_links_done + set_size

        predicted_links = predicted_links_list[i]

        for predicted_link, prediction in zip(predicted_links, predictions):
            eps = 10 ** -10
            likelihood = intercept+scaling*float(np.log10(prediction+eps)-np.log10(1-prediction+eps))
            scaled_prediction = (10**likelihood)/(1+10**likelihood)

            experiment.link_data.set_link_data(predicted_link[0], predicted_link[1], data_name="link_probability",
                                               value=float(scaled_prediction))
            experiment.link_data.set_link_data(predicted_link[0], predicted_link[1], data_name="link_penalty",
                                               value=float(-likelihood))

    # If predictions replace existing data, record overlap. Useful for evaluation purposes.
    if experiment.links is not None:
        for link in experiment.links.find_all_links():
            experiment.link_data.set_link_data(link[0], link[1], data_name="present_in_original",
                                               value=True)

    print("Saving file...")
    experiment.links = possible_links
    output_file = os.path.join(_output_folder, f"{experiment_index + 1}. {experiment.name.get_save_name()}."
                               + io.FILE_EXTENSION)
    io.save_data_to_json(experiment, output_file)
    experiments_to_save.append(experiment)

list_io.save_experiment_list_file(experiments_to_save,
                                  os.path.join(_output_folder, "_All" + list_io.FILES_LIST_EXTENSION))
print("Done!")
