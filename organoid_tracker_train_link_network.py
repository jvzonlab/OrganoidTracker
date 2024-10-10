#!/usr/bin/env python3

"""Script used to train the convolutional neural network, so that it can recognize nuclei in 3D images."""
import json
import os
import random
from functools import partial
from typing import Set, Tuple

import numpy as np
import tensorflow as tf
import tifffile
from tensorflow.python.data import Dataset

from organoid_tracker.linear_models.calculate_overlap import overlap_per_frame
from organoid_tracker.linear_models.logistic_regression import platt_scaling
from organoid_tracker.config import ConfigFile, config_type_image_shape, config_type_int, config_type_bool
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import io
from organoid_tracker.link_detection_cnn.convolutional_neural_network import build_model, tensorboard_callback
from organoid_tracker.link_detection_cnn.training_data_creator import create_image_with_links_list
from organoid_tracker.link_detection_cnn.training_dataset import training_data_creator_from_raw


# PARAMETERS

class _PerExperimentParameters:
    images_container: str
    images_pattern: str
    images_channels: Set[int]
    min_time_point: int
    max_time_point: int
    training_positions_file: str
    time_window_before: int
    time_window_after: int

    def to_experiment(self) -> Experiment:
        experiment = io.load_data_file(self.training_positions_file, self.min_time_point, self.max_time_point)
        general_image_loader.load_images(experiment, self.images_container, self.images_pattern,
                                         min_time_point=self.min_time_point, max_time_point=self.max_time_point)
        if self.images_channels != {1}:
            # Replace the first channel
            old_channels = experiment.images.get_channels()
            new_channels = [old_channels[index - 1] for index in self.images_channels]
            channel_merging_image_loader = ChannelSummingImageLoader(experiment.images.image_loader(), [new_channels])
            experiment.images.image_loader(channel_merging_image_loader)
        return experiment


print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("train_link_network")

per_experiment_params = []
i = 1
while True:
    params = _PerExperimentParameters()
    params.images_container = config.get_or_prompt(f"images_container_{i}",
                                                   "If you have a folder of image files, please paste the folder"
                                                   " path here. Else, if you have a LIF file, please paste the path to that file"
                                                   " here.")
    if params.images_container == "<stop>":
        break

    params.images_pattern = config.get_or_prompt(f"images_pattern_{i}",
                                                 "What are the image file names? (Use {time:03} for three digits"
                                                 " representing the time point, use {channel} for the channel)")
    channels_str = config.get_or_default(f"images_channels_{i}", "1", comment="What image channels are used? For"
                                                                              " example, use 1,2,4 to train on the sum of the 1st, 2nd and 4th channel.")
    params.images_channels = {int(part) for part in channels_str.split(",")}
    params.training_positions_file = config.get_or_default(f"positions_file_{i}",
                                                           f"positions_{i}.{io.FILE_EXTENSION}",
                                                           comment="What are the detected positions for those images?")
    params.min_time_point = int(config.get_or_default(f"min_time_point_{i}", str(0)))
    params.max_time_point = int(config.get_or_default(f"max_time_point_{i}", str(9999)))

    per_experiment_params.append(params)
    i += 1

time_window = [int(config.get_or_default(f"time_window_before", str(-1))),
               int(config.get_or_default(f"time_window_after", str(1)))]

use_TFR = config.get_or_default(f"use_tfrecords", str(False), type=config_type_bool)

patch_shape_zyx = list(
    config.get_or_default("patch_shape", "16, 32, 32", comment="Size in pixels (x, y, z) of the patches used"
                                                               " to train the network.",
                          type=config_type_image_shape))
print(patch_shape_zyx)
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
experiment_provider = (params.to_experiment() for params in per_experiment_params)

# Create a list of images and annotated positions
image_with_links_list = create_image_with_links_list(experiment_provider)

# shuffle training/validation data
random.seed("using a fixed seed to ensure reproducibility")
random.shuffle(image_with_links_list)

# save which frames will be used for validation so that we can do the platt scaling on these
validation_list = []
for image_with_links in image_with_links_list[-round(0.2*len(image_with_links_list)):]:
    validation_list.append((image_with_links.experiment_name, image_with_links.time_point.time_point_number()))

# get mean number of positions per timepoint
number_of_postions = []
for image_with_links in image_with_links_list:
    number_of_postions.append(image_with_links.xyz_positions.shape[0])
number_of_postions = np.mean(number_of_postions)

# create tf.datasets that generate the data
if use_TFR:
    raise ValueError("TFR not implemented...")
    # image_files, label_files, target_label_files, linked_files = dataset_writer(image_with_links_list, time_window, shards=10)

    # training_dataset = training_data_creator_from_TFR(image_files, label_files, target_label_files, linked_files,
    # patch_shape=patch_shape, batch_size=batch_size, mode='train',
    # split_proportion=0.8, n_images=len(image_with_links_list))
    # validation_dataset = training_data_creator_from_TFR(image_files, label_files, target_label_files, linked_files,
    # patch_shape=patch_shape, batch_size=batch_size,
    # mode='validation', split_proportion=0.8, n_images=len(image_with_links_list))

else:
    training_dataset = training_data_creator_from_raw(image_with_links_list, time_window=time_window,
                                                      patch_shape=patch_shape_zyx, batch_size=batch_size, mode='train',
                                                      split_proportion=0.8)
    validation_dataset = training_data_creator_from_raw(image_with_links_list, time_window=time_window,
                                                        patch_shape=patch_shape_zyx, batch_size=batch_size,
                                                        mode='validation', split_proportion=0.8)

model = build_model(shape=(patch_shape_zyx[0], patch_shape_zyx[1], patch_shape_zyx[2], time_window[1] - time_window[0] + 1),
                    batch_size=None)
model.summary()

print("Training...")
print(training_dataset)
history = model.fit(training_dataset,
                    epochs=epochs,
                    steps_per_epoch=round(0.8 * len(image_with_links_list) * 1 * number_of_postions / batch_size),
                    validation_data=validation_dataset,
                    validation_steps=round(0.2 * len(image_with_links_list) * 0.9 * number_of_postions / batch_size),
                    callbacks=[
                               tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)])

print("Saving model...")
trained_model_folder = os.path.join(output_folder, "model_links")
tf.keras.models.save_model(model, trained_model_folder)

# Perform Platt scaling
def predict(inputs, linked, model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor]:

    linked = linked['out']

    return model(inputs, training=False), linked

# create list with links without upsampling
experiment_provider = (params.to_experiment() for params in per_experiment_params)
list_for_platt_scaling =  create_image_with_links_list(experiment_provider, division_multiplier=1, mid_distance_multiplier=1)
# limit platt scaling to validation set
list_for_platt_scaling_val = []

for i in list_for_platt_scaling:
    pair = (i.experiment_name, i.time_point.time_point_number())
    if pair in validation_list:
        list_for_platt_scaling_val.append(i)

random.shuffle(list_for_platt_scaling_val)

callibration_dataset = training_data_creator_from_raw(list_for_platt_scaling_val, time_window=time_window,
                                                        patch_shape=patch_shape_zyx, batch_size=batch_size,
                                                        mode='validation', split_proportion=0.0, perturb_validation=False)
quick_dataset: Dataset = callibration_dataset.take(round(0.2 * len(image_with_links_list) * 1 * number_of_postions / batch_size))

outputs = quick_dataset.map(partial(predict, model=model))

prediction = []
linked = []
for i, element in enumerate(outputs):
    if len(element[0]>1):
        prediction = prediction + element[0].numpy().tolist()
        linked = linked + element[1].numpy().tolist()

prediction = np.array(prediction)
linked = np.array(linked)

(intercept, scaling, scaling_no_intercept) = platt_scaling(prediction, linked)
print('platt_scaling')
print(intercept)
print(scaling)

# save metadata model
with open(os.path.join(trained_model_folder, "settings.json"), "w") as file_handle:
    json.dump({"type": "links", "time_window": time_window, "patch_shape_zyx": patch_shape_zyx,
               "platt_intercept": intercept, "platt_scaling": scaling
               }, file_handle, indent=4)

# save validation list
with open(os.path.join(output_folder, "validation_list.json"), "w") as file_handle:
    json.dump(validation_list, file_handle, indent=4)

# # generate examples for reality check
def predict_with_input(inputs, linked, model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    image = inputs['input_1']
    target_image = inputs['input_2']
    distance = inputs['input_distances']

    linked = linked['out']
    return model(inputs, training=False), linked, image, target_image, distance

# Generate examples
os.makedirs(os.path.join(output_folder, "examples"), exist_ok=True)

quick_dataset: Dataset = validation_dataset.unbatch().take(1000).batch(1)

predictions = quick_dataset.map(partial(predict_with_input, model=model))

correct_examples = 0
incorrect_examples = 0

for i, element in enumerate(predictions):

    eps = 10 ** -10
    prediction = element[0]
    score = -np.log10(prediction + eps) + np.log10(1 - prediction + eps)

    linked = 2 * element[1].numpy() - 1

    image = element[2].numpy()
    image = image[0, :, :, :, :]
    #image = np.swapaxes(image, 0, -1)
    image = np.swapaxes(image, 1, -1)

    target_image = element[3].numpy()
    target_image = target_image[0, :, :, :, :]
    #target_image = np.swapaxes(target_image, 0, 2)
    #target_image = np.swapaxes(target_image, 0, -1)
    target_image = np.swapaxes(target_image, 1, -1)

    print(target_image.shape)

    if ((linked * score) < 0) and (correct_examples < 20):
        tifffile.imwrite(os.path.join(output_folder, "examples",
                                      "CORRECT_example_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score)) + ".ome.tiff"), image,imagej=True, metadata={'axes': 'TZXY'})
        print(element[4])
        tifffile.imwrite(os.path.join(output_folder, "examples",
                                      "CORRECT_example_target_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score)) + ".ome.tiff"), target_image,imagej=True, metadata={'axes': 'TZXY'})
        correct_examples = correct_examples + 1

    if ((linked * score) > 0) and (incorrect_examples < 20):
        tifffile.imwrite(os.path.join(output_folder, "examples",
                                      "INCORRECT_example_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score)) + ".ome.tiff"), image,imagej=True, metadata={'axes': 'TZXY'})
        print(element[4])
        distance = element[4].numpy()[0,:]
        tifffile.imwrite(os.path.join(output_folder, "examples",
                                      "INCORRECT_example_target_input" + str(i) + '_score_' +
                                      "{:.2f}".format(float(score))
                                      + '_x_' +"{:.2f}".format(float(distance[1]))
                                      + '_y_' +"{:.2f}".format(float(distance[2])) + ".ome.tiff"),
                         target_image, imagej=True, metadata={'axes': 'TZXY'})
        incorrect_examples = incorrect_examples + 1
    if (incorrect_examples == 10) and (correct_examples ==10):
        break