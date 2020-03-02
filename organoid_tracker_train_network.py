#!/usr/bin/env python3

"""Script used to train the convolutional neural network, so that it can recognize nuclei in 3D images."""
from os import path
import os

from organoid_tracker.config import ConfigFile, config_type_image_shape, config_type_int
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import io
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.position_detection_cnn import training_data_creator, trainer


# PARAMETERS
class _PerExperimentParameters:
    images_container: str
    images_pattern: str
    min_time_point: int
    max_time_point: int
    training_positions_file: str

    def to_experiment(self) -> Experiment:
        experiment = io.load_data_file(self.training_positions_file, self.min_time_point, self.max_time_point)
        general_image_loader.load_images(experiment, self.images_container, self.images_pattern,
                                         min_time_point=self.min_time_point, max_time_point=self.max_time_point)
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
    params.training_positions_file = config.get_or_default(f"positions_file_{i}",
                                                           f"positions_{i}.{io.FILE_EXTENSION}",
                                                           comment="What are the detected positions for those images?")
    params.min_time_point = int(config.get_or_default(f"min_time_point_{i}", str(0)))
    params.max_time_point = int(config.get_or_default(f"max_time_point_{i}", str(9999)))
    per_experiment_params.append(params)
    i += 1

patch_shape = config.get_or_default("patch_shape", "64, 64, 32", comment="Size in pixels (x, y, z) of the patches used"
                                                                         " to train the network.", type=config_type_image_shape)
image_shape = config.get_or_default("image_shape", "512, 512, 32", comment="Size in pixels (x, y, z) of all images."
                                    " Smaller images will be padded automatically, but larger images result in an"
                                    " error.", type=config_type_image_shape)
output_folder = config.get_or_default("output_folder", "training_output_folder", comment="Folder that will contain the"
                                      " trained model.")
batch_size = config.get_or_default("batch_size", "64", comment="How many patches are used for training at once. A"
                                                               " higher batch size can load to a better training"
                                                               " result.", type=config_type_int)
max_training_steps = config.get_or_default("max_training_steps", "100000", comment="For how many iterations the network"
                                           " is trained. Larger is not always better; at some point the network might"
                                           " get overfitted to your training data.", type=config_type_int)
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Starting...")
# Create a generator that will load the experiments on demand
experiment_provider = (params.to_experiment() for params in per_experiment_params)

print("Note: this script can easily take several hours or even days to complete.")
print("Creating training files...")
checkpoint_dir = path.join(output_folder, "checkpoints")
tfrecord_dir = path.join(output_folder, "tfrecord")
if not path.exists(tfrecord_dir):
    training_data_creator.create_training_data(experiment_provider, out_dir=tfrecord_dir, image_size_zyx=image_shape)
else:
    print("   Skipped! Already found a tfrecord directory, assuming it's good.")

print("Training...")
trainer.train(tfrecord_dir, checkpoint_dir, patch_size_zyx=patch_shape, image_size_zyx=image_shape,
              batch_size=batch_size, max_steps=max_training_steps, use_cpu_output=False)

print("")
print("Done! Was it worth the wait?")
print("To get an insight in the training process, run Tensorboard:")
print("    tensorboard --logdir=\"" + path.abspath(checkpoint_dir) + "\"")
