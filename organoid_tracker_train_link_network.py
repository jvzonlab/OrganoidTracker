#!/usr/bin/env python3

"""Script used to train the convolutional neural network, so that it can recognize the same nucleus across time points.
"""
import _keras_environment
_keras_environment.activate()

import json
import os
import random
from typing import Tuple

import keras.callbacks
import keras.saving
import numpy
import numpy as np
import tifffile
from torch.utils.data import DataLoader

from organoid_tracker.config import ConfigFile, config_type_image_shape_xyz_to_zyx, config_type_int
from organoid_tracker.imaging import list_io
from organoid_tracker.linear_models.logistic_regression import platt_scaling
from organoid_tracker.neural_network.dataset_transforms import LimitingDataset
from organoid_tracker.neural_network.link_detection_cnn.convolutional_neural_network import build_model
from organoid_tracker.neural_network.link_detection_cnn.training_data_creator import create_image_with_links_list
from organoid_tracker.neural_network.link_detection_cnn.training_dataset import training_data_creator_from_raw

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("train_link_network")
dataset_file = config.get_or_prompt("dataset_file", "Please paste the path here to the dataset file."
                                                    " You can generate such a file from OrganoidTracker using File -> Tabs -> "
                                                    " all tabs.", store_in_defaults=True)
time_window = (int(config.get_or_default(f"time_window_before", str(-1))),
               int(config.get_or_default(f"time_window_after", str(1))))

patch_shape_zyx: Tuple[int, int, int] = tuple(
    config.get_or_default("patch_shape", "32, 32, 16", comment="Size in pixels (x, y, z) of the patches used"
                                                               " to train the network.",
                          type=config_type_image_shape_xyz_to_zyx))
output_folder = config.get_or_default("output_folder", "training_output_folder", comment="Folder that will contain the"
                                                                                         " trained model.")
batch_size = config.get_or_default("batch_size", "64", comment="How many patches are used for training at once. A"
                                                               " higher batch size can load to a better training"
                                                               " result.", type=config_type_int)
epochs = config.get_or_default("epochs", "50", comment="For how many epochs the network is trained. Larger is not"
                                                       " always better; at some point the network might get overfitted"
                                                       " to your training data.",
                               type=config_type_int)

config.save_and_exit_if_changed()
# END OF PARAMETERS

print('creating list of links')
# Create a generator that will load the experiments on demand
experiment_provider = list_io.load_experiment_list_file(dataset_file)

# Create a list of images and annotated positions
image_with_links_list = create_image_with_links_list(experiment_provider)

# shuffle training/validation data
random.seed("using a fixed seed to ensure reproducibility")
random.shuffle(image_with_links_list)

# save which frames will be used for validation so that we can do the platt scaling on these
os.makedirs(output_folder, exist_ok=True)
validation_list = []
for image_with_links in image_with_links_list[-round(0.2 * len(image_with_links_list)):]:
    validation_list.append((image_with_links.experiment_name, image_with_links.time_point.time_point_number()))

# get mean number of positions per timepoint
number_of_postions = []
for image_with_links in image_with_links_list:
    number_of_postions.append(image_with_links.xyz_positions.shape[0])
number_of_postions = np.mean(number_of_postions)

# create datasets that generate the data

training_dataset = training_data_creator_from_raw(image_with_links_list, time_window=time_window,
                                                  patch_shape=patch_shape_zyx, batch_size=batch_size, mode='train',
                                                  split_proportion=0.8)
validation_dataset = training_data_creator_from_raw(image_with_links_list, time_window=time_window,
                                                    patch_shape=patch_shape_zyx, batch_size=batch_size,
                                                    mode='validation', split_proportion=0.8)

debug_sample = next(iter(training_dataset))
print(debug_sample)

model = build_model(
    shape=(patch_shape_zyx[0], patch_shape_zyx[1], patch_shape_zyx[2], time_window[1] - time_window[0] + 1),
    batch_size=None)
model.summary()

print("Training...")
trained_model_folder = os.path.join(output_folder, "model_links")
logging_folder = os.path.join(trained_model_folder, "training_logging")
os.makedirs(logging_folder, exist_ok=True)
history = model.fit(training_dataset,
                    epochs=epochs,
                    steps_per_epoch=len(training_dataset),
                    validation_data=validation_dataset,
                    validation_steps=len(validation_dataset),
                    callbacks=[
                        keras.callbacks.CSVLogger(os.path.join(logging_folder, "logging.csv"), separator=",",
                                                  append=False),
                        keras.callbacks.EarlyStopping(patience=1, min_delta=0.001, restore_best_weights=True)])

print("Saving model...")
os.makedirs(trained_model_folder, exist_ok=True)
model.save(os.path.join(trained_model_folder, "model.keras"))

# Perform Platt scaling
print("Performing Platt scaling...")

# new list without any upsampling, based on validation list
experiment_provider = list_io.load_experiment_list_file(dataset_file)
list_for_platt_scaling = create_image_with_links_list(experiment_provider, division_multiplier=1,
                                                      mid_distance_multiplier=1)

# limit platt scaling to validation set
list_for_platt_scaling_val = []

for i in list_for_platt_scaling:
    pair = (i.experiment_name, i.time_point.time_point_number())
    if pair in validation_list:
        list_for_platt_scaling_val.append(i)

random.shuffle(list_for_platt_scaling_val)

calibration_dataset = training_data_creator_from_raw(list_for_platt_scaling_val, time_window=time_window,
                                                     patch_shape=patch_shape_zyx, batch_size=batch_size,
                                                     mode='validation', split_proportion=0.0, perturb=False)
calibration_dataset = DataLoader(
    LimitingDataset(calibration_dataset.dataset, round(0.2 * len(image_with_links_list) * 1 * number_of_postions)),
    batch_size=batch_size)

predicted_chances_all = []
ground_truth_linked = []
for sample in calibration_dataset:
    output_element = model.predict(sample[0], verbose=0)
    predicted_chances_all += np.squeeze(output_element).tolist()
    ground_truth_linked += keras.ops.convert_to_numpy(sample[1]).tolist()

predicted_chances_all = np.array(predicted_chances_all)
ground_truth_linked = np.array(ground_truth_linked)

(intercept, scaling, scaling_no_intercept) = platt_scaling(predicted_chances_all, ground_truth_linked)
print(f'Result: y = 1 / (1 + exp(-({scaling:.2f} * x + {intercept:.2f})))')

# save metadata model
with open(os.path.join(trained_model_folder, "settings.json"), "w") as file_handle:
    json.dump({"type": "links", "time_window": time_window, "patch_shape_zyx": patch_shape_zyx,
               "platt_intercept": intercept, "platt_scaling": scaling
               }, file_handle, indent=4)

# save validation list
with open(os.path.join(output_folder, "validation_list.json"), "w") as file_handle:
    json.dump(validation_list, file_handle, indent=4)

# Generate examples
link_examples_folder = os.path.join(output_folder, "link_examples")
os.makedirs(link_examples_folder, exist_ok=True)

quick_dataset = DataLoader(LimitingDataset(validation_dataset.dataset, 1000), batch_size=1)

predictions = model.predict(quick_dataset)

correct_examples = 0
incorrect_examples = 0

for sample in quick_dataset:
    # We use a batch size of 1, so we can just take the first element
    input_element = sample[0]
    output_element = model.predict(input_element, verbose=0)

    predicted_chance = numpy.squeeze(output_element)

    eps = 10 ** -10
    score = -np.log10(predicted_chance + eps) + np.log10(1 - predicted_chance + eps)

    ground_truth_linked = 2 * keras.ops.convert_to_numpy(sample[1]) - 1

    image = keras.ops.convert_to_numpy(input_element[0])
    image = image[0, :, :, :, :]
    image = np.swapaxes(image, 1, -1)

    target_image = keras.ops.convert_to_numpy(input_element[1])
    target_image = target_image[0, :, :, :, :]
    target_image = np.swapaxes(target_image, 1, -1)

    if ((ground_truth_linked * score) < 0) and (correct_examples < 20):
        tifffile.imwrite(os.path.join(link_examples_folder,
                                      "CORRECT_example_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score)) + ".ome.tiff"), image, imagej=True,
                         metadata={'axes': 'TZYX'})
        tifffile.imwrite(os.path.join(link_examples_folder,
                                      "CORRECT_example_target_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score)) + ".ome.tiff"), target_image, imagej=True,
                         metadata={'axes': 'TZYX'})
        correct_examples = correct_examples + 1

    if ((ground_truth_linked * score) > 0) and (incorrect_examples < 20):
        tifffile.imwrite(os.path.join(link_examples_folder,
                                      "INCORRECT_example_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score)) + ".ome.tiff"), image, imagej=True,
                         metadata={'axes': 'TZYX'})
        distance = keras.ops.convert_to_numpy(input_element[2])[0, :]
        tifffile.imwrite(os.path.join(link_examples_folder,
                                      "INCORRECT_example_target_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score))
                                      + '_x_' + "{:.2f}".format(float(distance[1]))
                                      + '_y_' + "{:.2f}".format(float(distance[2])) + ".ome.tiff"),
                         target_image, imagej=True, metadata={'axes': 'TZYX'})
        incorrect_examples = incorrect_examples + 1
    if (incorrect_examples == 10) and (correct_examples == 10):
        break
