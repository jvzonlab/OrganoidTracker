#!/usr/bin/env python3

"""Script used to train the convolutional neural network, so that it can recognize nuclei in 3D images."""
import os
import random
from functools import partial
from os import path
from typing import Set

import tensorflow as tf
from PIL import Image as Img
from tifffile import tifffile

from organoid_tracker.config import ConfigFile, config_type_image_shape, config_type_int
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import io
# from organoid_tracker.position_detection_cnn import training_data_creator, trainer

from organoid_tracker.position_detection_cnn.convolutional_neural_network import build_model, tensorboard_callback
from organoid_tracker.position_detection_cnn.training_data_creator import create_image_with_positions_list

from organoid_tracker.position_detection_cnn.training_dataset import  training_data_creator_from_TFR, training_data_creator_from_raw


# PARAMETERS
from organoid_tracker_train_network import tensorboard_folder


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
config = ConfigFile("train_network")

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

use_TFR = config.get_or_default(f"use_TFRecords", str(True))
use_TFR = bool(use_TFR == "True")

patch_shape = list(
    config.get_or_default("patch_shape", "64, 64, 32", comment="Size in pixels (x, y, z) of the patches used"
                                                               " to train the network.",
                          type=config_type_image_shape))

output_folder = config.get_or_default("output_folder", "training_output_folder", comment="Folder that will contain the"
                                                                                         " trained model.")
batch_size = config.get_or_default("batch_size", "64", comment="How many patches are used for training at once. A"
                                                               " higher batch size can load to a better training"
                                                               " result.", type=config_type_int)
max_training_steps = config.get_or_default("max_training_steps", "100000", comment="For how many iterations the network"
                                                                                   " is trained. Larger is not always better; at some point the network might"
                                                                                   " get overfitted to your training data.",
                                           type=config_type_int)
config.save_and_exit_if_changed()
# END OF PARAMETERS


print('waddup')

# Create a generator that will load the experiments on demand
experiment_provider = (params.to_experiment() for params in per_experiment_params[:-1])

# Create a list of images and annotated positions
image_with_positions_list = create_image_with_positions_list(experiment_provider)

# shuffle training/validation data
random.seed("using a fixed seed to ensure reproducibility")
random.shuffle(image_with_positions_list)

training_dataset = training_data_creator_from_raw(image_with_positions_list, time_window=time_window,
                                                  patch_shape=patch_shape, batch_size=batch_size, mode='train',
                                                  split_proportion=1)


experiment_provider = (params.to_experiment() for params in per_experiment_params[-1:])

# Create a list of images and annotated positions
image_with_positions_list_val = create_image_with_positions_list(experiment_provider)
random.shuffle(image_with_positions_list_val)

validation_dataset = training_data_creator_from_raw(image_with_positions_list_val, time_window=time_window,
                                            patch_shape=patch_shape, batch_size=batch_size,
                                            mode='validation', split_proportion=0.0)

print("Defining model...")
model = build_model(shape=(patch_shape[0], None, None, time_window[1] - time_window[0] + 1), batch_size=None)
model.summary()

print("Trainung...")
history = model.fit(training_dataset,
                    epochs=4,
                    steps_per_epoch=round(0.8*len(image_with_positions_list)),
                    validation_data=validation_dataset,
                    validation_steps=10,
                    callbacks=[tensorboard_callback(tensorboard_folder)])

print("Saving model...")
tf.keras.models.save_model(model, "model_test")


# Sanity check, do predictions on 10 samples of the validation set
print("Sanity check...")


def predict(image, label, model=model):
    return image, model(image, training=False), label


quick_dataset = validation_dataset.unbatch().take(10).batch(1)
predictions = quick_dataset.map(partial(predict, model=model))

iterator = iter(predictions)

for i in range(10):
    element = iterator.get_next()

    array = element[0].numpy()
    array = array[0, :, :, :, 0]
    tifffile.imsave("example_input" + str(i) + ".tiff", array, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

    array = element[1].numpy()
    array = array[0, :, :, :, 0]
    tifffile.imsave("example_prediction" + str(i) + ".tiff", array, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})

    array = element[2].numpy()
    array = array[0, :, :, :, 0]
    tifffile.imsave("input_prediction" + str(i) + ".tiff", array, compression=tifffile.COMPRESSION.ADOBE_DEFLATE, compressionargs={"level": 9})


