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


if " " in os.getcwd():
    print(f"Unfortunately, we cannot train the neural network in a folder that contains spaces in its path."
          f" So '{os.getcwd()}' is not a valid location.")
    exit()

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

patch_shape = list(
    config.get_or_default("patch_shape", "64, 64, 32", comment="Size in pixels (x, y, z) of the patches used"
                                                               " to train the network.",
                          type=config_type_image_shape))

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
                                                      patch_shape=patch_shape, batch_size=batch_size, mode='train',
                                                      split_proportion=0.8)
    validation_dataset = training_data_creator_from_raw(image_with_links_list, time_window=time_window,
                                                        patch_shape=patch_shape, batch_size=batch_size,
                                                        mode='validation', split_proportion=0.8)

model = build_model(shape=(patch_shape[0], patch_shape[1], patch_shape[2], time_window[1] - time_window[0] + 1),
                    batch_size=None)
model.summary()

print("Training...")
print(training_dataset)
history = model.fit(training_dataset,
                    epochs=epochs,
                    steps_per_epoch=round(0.8 * len(image_with_links_list) * 1 * number_of_postions / batch_size),
                    validation_data=validation_dataset,
                    validation_steps=round(0.2 * len(image_with_links_list) * 1 * number_of_postions / batch_size),
                    callbacks=[tensorboard_callback,
                               tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

print("Saving model...")
trained_model_folder = os.path.join(output_folder, "model_links")
tf.keras.models.save_model(model, trained_model_folder)
with open(os.path.join(trained_model_folder, "settings.json"), "w") as file_handle:
    json.dump({"type": "links", "time_window": time_window, "patch_shape_xyz": patch_shape}, file_handle, indent=4)


# generate examples for reality check
def predict(data, model: tf.keras.Model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    inputs = data[0]

    image = inputs['input_1']
    target_image = inputs['input_2']
    linked = data[1]

    return model(inputs, training=False), linked, image, target_image


quick_dataset: Dataset = validation_dataset.unbatch().take(1000).batch(1)

predictions = quick_dataset.map(partial(predict, model=model))

correct_examples = 0
incorrect_examples = 0

for i, element in enumerate(predictions):

    eps = 10 ** -10
    prediction = element[0]
    score = -np.log10(prediction + eps) + np.log10(1 - prediction + eps)

    linked = 2 * element[1] - 1

    image = element[2].numpy()
    image = image[0, :, :, :, :]
    image = np.swapaxes(image, 2, -1)
    image = np.swapaxes(image, -2, -1)

    target_image = element[2].numpy()
    target_image = target_image[0, :, :, :, :]
    target_image = np.swapaxes(target_image, 2, -1)
    target_image = np.swapaxes(target_image, -2, -1)

    if ((linked * score) < 0) and (correct_examples < 10):
        tifffile.imwrite(os.path.join(output_folder, "examples",
                                      "CORRECT_example_input" + str(i) + '_score_' +
                                      "{:.2f}".format(score) + ".tiff"), image)
        tifffile.imwrite(os.path.join(output_folder, "examples",
                                      "CORRECT_example_target_input" + str(i) + '_score_' +
                                      "{:.2f}".format(score) + ".tiff"), target_image)
        correct_examples = correct_examples + 1

    if ((linked * score) > 0) and (incorrect_examples < 10):
        tifffile.imwrite(os.path.join(output_folder, "examples",
                                      "CORRECT_example_input" + str(i) + '_score_' +
                                      "{:.2f}".format(score) + ".tiff"), image)
        tifffile.imwrite(os.path.join(output_folder, "examples",
                                      "CORRECT_example_target_input" + str(i) + '_score_' +
                                      "{:.2f}".format(score) + ".tiff"), target_image)
        incorrect_examples = incorrect_examples + 1
