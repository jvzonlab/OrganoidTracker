#!/usr/bin/env python3

import json
import os
import random
import numpy as np

from typing import Set, Iterable
from organoid_tracker.config import ConfigFile
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import io
from organoid_tracker.linear_models.logistic_regression import platt_scaling
from organoid_tracker.local_marginalization.local_marginalization_functions import local_marginalization, \
    minimal_marginalization


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

def create_experiment_with_links_list(experiments: Iterable[Experiment]):

    experiment_with_links_list = []

    for experiment in experiments:

        for time_point in experiment.positions.time_points():
            # read a single time point
            links = experiment.links.of_time_point(time_point)

            experiment_with_links_list.append((experiment, links, time_point))

    return experiment_with_links_list

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("callibration")

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

_steps = int(config.get_or_default("size subset (steps away from link of interest)", str(3)))

config.save_and_exit_if_changed()
# END OF PARAMETERS

# Create a generator that will load the experiments on demand
experiment_provider = (params.to_experiment() for params in per_experiment_params)

# Create a list of images and annotated links
experiment_with_links_list = create_experiment_with_links_list(experiment_provider)

# shuffle training/validation data
random.seed("using a fixed seed to ensure reproducibility")
random.shuffle(experiment_with_links_list)

# Marginalization
marginal_predictions = []
predictions = []
correct_links = []

number_of_frames = len(experiment_with_links_list)
index = 0

for (experiment, links, time) in experiment_with_links_list:

    print(f"Marginalizing frame {index}/{number_of_frames}")

    for (position1, position2) in links:

        if abs(experiment.link_data.get_link_data(position1, position2, data_name="link_penalty")) < 4:
            marginal_predictions.append(local_marginalization(position1, position2, experiment, steps=_steps, complete_graph=True))
        else:
            marginal_predictions.append(minimal_marginalization(position1, position2, experiment))

        correct = experiment.link_data.get_link_data(position1, position2, data_name="present_in_original")
        if correct is None:
            correct = False

        correct_links.append(correct)

(intercept, scaling, scaling_no_intercept) = platt_scaling(np.array(marginal_predictions), np.array(correct_links))
print('temperature:')
print(1/scaling_no_intercept)

with open("callibration.json", "w") as file_handle:
    json.dump({"steps": _steps, "platt_scaling": scaling_no_intercept, "temperature": 1/scaling_no_intercept
               }, file_handle, indent=4)



