#!/usr/bin/env python3

"""Script used to train the convolutional neural network, so that it can recognize nuclei in 3D images."""
import json
import os
import random
from functools import partial
from typing import Set, Tuple

os.environ["KERAS_BACKEND"] = "torch"
import keras
import keras.callbacks
import keras.models
import tifffile
from torch import Tensor

from organoid_tracker.config import ConfigFile, config_type_image_shape, config_type_int
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.image_loading.builtin_merging_image_loaders import ChannelSummingImageLoader
from organoid_tracker.imaging import io
from organoid_tracker.neural_network.position_detection_cnn.convolutional_neural_network import build_model
from organoid_tracker.neural_network.position_detection_cnn.custom_filters import distance_map
from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import \
    create_image_with_positions_list
from organoid_tracker.neural_network.position_detection_cnn.training_dataset import training_data_creator_from_raw
from organoid_tracker.neural_network.position_detection_cnn.training_inspection_callback import WriteExamplesCallback, \
    ExampleDataset


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
config = ConfigFile("train_position_network")

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

time_window = (int(config.get_or_default(f"time_window_before", str(-1))),
               int(config.get_or_default(f"time_window_after", str(1))))

patch_shape_zyx = list(
    config.get_or_default("patch_shape", "32, 64, 64", comment="Size in pixels (z, y, x) of the patches used"
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

# Create a generator that will load the experiments on demand
experiment_provider = (params.to_experiment() for params in per_experiment_params)

# Create a list of images and annotated positions
image_with_positions_list = create_image_with_positions_list(experiment_provider)

# shuffle training/validation data
seed = 42
random = random.Random(seed)
random.shuffle(image_with_positions_list)

# create datasets that generate the data
training_dataset = training_data_creator_from_raw(image_with_positions_list, time_window=time_window, seed=seed,
                                                  patch_shape=patch_shape_zyx, batch_size=batch_size, mode='train',
                                                  split_proportion=0.8, crop=True)
validation_dataset = training_data_creator_from_raw(image_with_positions_list, time_window=time_window, seed=seed,
                                                    patch_shape=patch_shape_zyx, batch_size=batch_size,
                                                    mode='validation', split_proportion=0.8, crop=True)


print("Defining model...")
model = build_model(shape=(patch_shape_zyx[0], None, None, time_window[1] - time_window[0] + 1), batch_size=None)
model.summary()

print("Training...")
logging_folder = os.path.join(output_folder, "training_logging")
os.makedirs(logging_folder, exist_ok=True)
example_x, example_y_true = next(iter(validation_dataset))
example_dataset = ExampleDataset(input=keras.ops.convert_to_numpy(example_x), y_true=keras.ops.convert_to_numpy(example_y_true))
del example_x, example_y_true

history = model.fit(training_dataset,
                    epochs=epochs,
                    steps_per_epoch=len(training_dataset),
                    validation_data=validation_dataset,
                    validation_steps=len(validation_dataset),
                    callbacks=[
                        keras.callbacks.CSVLogger(os.path.join(logging_folder, "logging.csv"), separator=",", append=False),
                        WriteExamplesCallback(logging_folder, example_dataset),
                        keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)])


print("Saving model...")
trained_model_folder = os.path.join(output_folder, "model_positions")
os.makedirs(trained_model_folder, exist_ok=True)
model.save(os.path.join(trained_model_folder, "model.keras"))
with open(os.path.join(trained_model_folder, "settings.json"), "w") as file_handle:
    json.dump({"type": "positions", "time_window": time_window}, file_handle, indent=4)
print("Done! Model is in " + os.path.abspath(trained_model_folder))
