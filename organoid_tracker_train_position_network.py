#!/usr/bin/env python3

"""Script used to train the convolutional neural network, so that it can recognize nuclei in 3D images."""
import _keras_environment
_keras_environment.activate()

import json
import os
import random

import keras.callbacks
import keras.models

from organoid_tracker.config import ConfigFile, config_type_image_shape_xyz_to_zyx, config_type_int
from organoid_tracker.imaging import list_io
from organoid_tracker.neural_network.position_detection_cnn.convolutional_neural_network import build_model
from organoid_tracker.neural_network.position_detection_cnn.training_data_creator import \
    create_image_with_positions_list
from organoid_tracker.neural_network.position_detection_cnn.training_dataset import training_data_creator_from_raw
from organoid_tracker.neural_network.position_detection_cnn.training_inspection_callback import WriteExamplesCallback, \
    ExampleDataset


# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("train_position_network")
dataset_file = config.get_or_prompt("dataset_file", "Please paste the path here to the dataset file."
                                     " You can generate such a file from OrganoidTracker using File -> Tabs -> "
                                     " all tabs.", store_in_defaults=True)

time_window = (int(config.get_or_default(f"time_window_before", str(-1))),
               int(config.get_or_default(f"time_window_after", str(1))))

patch_shape_zyx = list(
    config.get_or_default("patch_shape", "64, 64, 32", comment="Size in pixels (x, y, z) of the patches used"
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

# Create a generator that will load the experiments on demand
experiment_provider = list_io.load_experiment_list_file(dataset_file)

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
trained_model_folder = os.path.join(output_folder, "model_positions")
logging_folder = os.path.join(trained_model_folder, "training_logging")
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
os.makedirs(trained_model_folder, exist_ok=True)
model.save(os.path.join(trained_model_folder, "model.keras"))
with open(os.path.join(trained_model_folder, "settings.json"), "w") as file_handle:
    json.dump({"type": "positions", "time_window": time_window}, file_handle, indent=4)
print("Done! Model is in " + os.path.abspath(trained_model_folder))
