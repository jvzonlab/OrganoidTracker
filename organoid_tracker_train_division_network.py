#!/usr/bin/env python3

"""Script used to train the convolutional neural network, so that it can recognize nuclei in 3D images."""
import _keras_environment
_keras_environment.activate()

import json
import os
import random

import keras.callbacks
import keras.saving
import numpy as np
import tifffile
from torch.utils.data import DataLoader

from organoid_tracker.config import ConfigFile, config_type_image_shape_xyz_to_zyx, config_type_int
from organoid_tracker.imaging import list_io
from organoid_tracker.linear_models.logistic_regression import platt_scaling
from organoid_tracker.neural_network.dataset_transforms import LimitingDataset
from organoid_tracker.neural_network.division_detection_cnn.convolutional_neural_network import build_model
from organoid_tracker.neural_network.division_detection_cnn.training_data_creator import \
    create_image_with_divisions_list
from organoid_tracker.neural_network.division_detection_cnn.training_dataset import training_data_creator_from_raw

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("train_division_network")
dataset_file = config.get_or_prompt("dataset_file", "Please paste the path here to the dataset file."
                                     " You can generate such a file from OrganoidTracker using File -> Tabs -> "
                                     " all tabs.", store_in_defaults=True)
full_window = bool(config.get_or_default(f"identify full division window", 'True'))

time_window = [int(config.get_or_default(f"time_window_before", str(-1))),
               int(config.get_or_default(f"time_window_after", str(1)))]

patch_shape_zyx = \
    config.get_or_default("patch_shape", "32, 32, 16", comment="Size in pixels (x, y, z) of the patches used"
                                                               " to train the network.",
                          type=config_type_image_shape_xyz_to_zyx)

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

# Create a generator that will load the experiments on demand
experiment_provider = list_io.load_experiment_list_file(dataset_file)

# Create a list of images and annotated positions
image_with_divisions_list = create_image_with_divisions_list(experiment_provider, full_window=full_window)

# shuffle training/validation data
random.seed("using a fixed seed to ensure reproducibility")
random.shuffle(image_with_divisions_list)

# save which frames will be used for validation so that we can do the platt scaling on these
validation_list = []
for image_with_divisions in image_with_divisions_list[-round(0.2*len(image_with_divisions_list)):]:
    validation_list.append((image_with_divisions.experiment_name, image_with_divisions.time_point.time_point_number()))

# create tf.datasets that generate the data
training_dataset = training_data_creator_from_raw(image_with_divisions_list, time_window=time_window,
                                                  patch_shape=patch_shape_zyx, batch_size=batch_size, mode='train',
                                                  split_proportion=0.8)
validation_dataset = training_data_creator_from_raw(image_with_divisions_list, time_window=time_window,
                                                    patch_shape=patch_shape_zyx, batch_size=batch_size,
                                                    mode='validation', split_proportion=0.8)

# build model
model = build_model(
    shape=(patch_shape_zyx[0], patch_shape_zyx[1], patch_shape_zyx[2], time_window[1] - time_window[0] + 1),
    batch_size=None)
model.summary()

# train model
print("Training...")
trained_model_folder = os.path.join(output_folder, "model_divisions")
logging_folder = os.path.join(trained_model_folder, "training_logging")
os.makedirs(logging_folder, exist_ok=True)

history = model.fit(training_dataset,
                    epochs=epochs,
                    steps_per_epoch=len(training_dataset),
                    validation_data=validation_dataset,
                    validation_steps=len(validation_dataset),
                    callbacks=[keras.callbacks.CSVLogger(os.path.join(logging_folder, "logging.csv"), separator=",", append=False),
                               keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

# save model
print("Saving model...")
os.makedirs(trained_model_folder, exist_ok=True)
model.save(os.path.join(trained_model_folder, "model.keras"))

# Perform Platt scaling
print("Performing Platt scaling...")

# new list without any upsampling, based on validation list
experiment_provider = list_io.load_experiment_list_file(dataset_file)
list_for_platt_scaling = create_image_with_divisions_list(experiment_provider, division_multiplier=1,
                                                          loose_end_multiplier=0, counter_examples_per_div=1000,
                                                          window=(0, 0))
list_for_platt_scaling_val = []
for i in list_for_platt_scaling:
    pair = (i.experiment_name, i.time_point.time_point_number())
    if pair in validation_list:
        list_for_platt_scaling_val.append(i)

calibration_dataset = training_data_creator_from_raw(list_for_platt_scaling_val, time_window=time_window,
                                                     patch_shape=patch_shape_zyx, batch_size=batch_size,
                                                     mode='validation', split_proportion=0.0, perturb=False)
calibration_dataset = DataLoader(LimitingDataset(calibration_dataset.dataset, max_samples=5000 * batch_size), batch_size=batch_size)

predicted_chances_all = []
ground_truth_dividing = []
for sample in calibration_dataset:
    element = model.predict(sample[0], verbose=0)
    predicted_chances_all += np.squeeze(element).tolist()
    ground_truth_dividing += keras.ops.convert_to_numpy(sample[1]).tolist()


predicted_chances_all = np.array(predicted_chances_all)
ground_truth_dividing = np.array(ground_truth_dividing)

(intercept, scaling, scaling_no_intercept) = platt_scaling(predicted_chances_all, ground_truth_dividing)
print(f'Result: y = 1 / (1 + exp(-({scaling:.2f} * x + {intercept:.2f})))')

# save metadata
with open(os.path.join(trained_model_folder, "settings.json"), "w") as file_handle:
    json.dump({"type": "divisions", "time_window": time_window, "patch_shape_zyx": patch_shape_zyx,
               "platt_intercept": intercept, "platt_scaling": scaling}, file_handle, indent=4)

# saves which timepoints are used for validation vs training
with open(os.path.join(output_folder, "validation_list.json"), "w") as file_handle:
    json.dump(validation_list, file_handle, indent=4)

# Generate examples for sanity check
print("Generating examples for sanity check...")
divisions_example_folder = os.path.join(output_folder, "division_examples")
os.makedirs(divisions_example_folder, exist_ok=True)


quick_dataset: DataLoader = DataLoader(validation_dataset.dataset, batch_size=1)

correct_examples = 0
incorrect_examples = 0

for i, element in enumerate(quick_dataset):
    predictions = model.predict(element[0], verbose=0)

    eps = 10 ** -10
    predicted_chances_all = np.squeeze(predictions[0])
    score = -np.log10(predicted_chances_all + eps) + np.log10(1 - predicted_chances_all + eps)

    ground_truth_dividing = 2 * keras.ops.convert_to_numpy(element[1]) - 1

    image = keras.ops.convert_to_numpy(element[0])
    image = image[0, :, :, :, :]

    # Order is now Z, Y, X, T, so move T to the start
    image = np.moveaxis(image, source=-1, destination=0)

    if ((ground_truth_dividing * score) < 0) and (correct_examples < 10):
        tifffile.imwrite(os.path.join(divisions_example_folder,
                                      "CORRECT_example_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score)) + ".ome.tiff"), image, imagej=True,
                         metadata={'axes': 'TZYX'})

        correct_examples = correct_examples + 1

    if ((ground_truth_dividing * score) > 0) and (incorrect_examples < 10):
        tifffile.imwrite(os.path.join(divisions_example_folder,
                                      "INCORRECT_example_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score)) + ".ome.tiff"), image, imagej=True,
                         metadata={'axes': 'TZYX'})

        incorrect_examples = incorrect_examples + 1

    if (incorrect_examples == 10) and (correct_examples == 10):
        break
