"""Predictions which positions are about to divide using an already-trained convolutional neural network."""
import _keras_environment
_keras_environment.activate()

import json
import os

import numpy as np

from organoid_tracker.config import ConfigFile, config_type_int, config_type_float
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.imaging import io, list_io
from organoid_tracker.neural_network.division_detection_cnn.division_predictor import load_division_model, \
    remove_division_oversegmentation

# PARAMETERS

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_divisions")
_dataset_file = config.get_or_prompt("dataset_file", "Please paste the path here to the dataset file."
                                     " You can generate such a file from OrganoidTracker using File -> Tabs -> "
                                     " all tabs.", store_in_defaults=True)


_model_folder = config.get_or_prompt("model_folder", "Please paste the path here to the \"trained_model\" folder containing the trained model.")
_output_folder = config.get_or_default("divisions_output_folder", "Division predictions", comment="Output folder for the divisions, can be viewed using the visualizer program.")
_channels_str = config.get_or_default("images_channels", str(1), comment="Index(es) of the channels to use. Use \"3\" to use the third channel for predictions. Use \"1,3,4\" to use the sum of the first, third and fourth channel for predictions.")
_images_channels = {ImageChannel(index_one=int(part)) for part in _channels_str.split(",")}
_batch_size = config.get_or_default("batch_size", str(64), type=config_type_int, comment="Batch size for predictions. If you run out of memory, lower this value. Increasing it will speed up predictions slightly (but won't affect the results).")
_min_distance_dividing = float(config.get_or_default("minimum_distance_between_dividing_cell (in micron)", str(4.5)))
_scale_factor_xy = config.get_or_default("scale_factor_xy", str(1.0), comment="Scale factor in x and y direction. A value of 0.5 will cause all images to be scaled down to half their size before being passed to the model.", type=config_type_float)
_scale_factor_z = config.get_or_default("scale_factor_z", str(1.0), comment="Scale factor in z direction.", type=config_type_float)
_intensity_quantile_min = config.get_or_default("intensity_min_quantile", str(0.01), comment="Minimum quantile for intensity normalization. Applied to entire 3D stack of each time point. A value of 0.0 means the minimum intensity is used.", type=config_type_float)
_intensity_quantile_max = config.get_or_default("intensity_max_quantile", str(0.99), comment="Maximum quantile for intensity normalization. A value of 1.0 means the maximum intensity is used.", type=config_type_float)

config.save()
# END OF PARAMETERS

# set relevant parameters
_model_folder = os.path.abspath(_model_folder)
with open(os.path.join(_model_folder, "settings.json")) as file_handle:
    json_contents = json.load(file_handle)
    if json_contents["type"] != "divisions":
        print("Error: model at " + _model_folder + " is made for working with " + str(json_contents["type"]) + ", not divisions")
        exit(1)
    time_window = json_contents["time_window"]
    patch_shape = json_contents["patch_shape_zyx"]

    scaling = json_contents["platt_scaling"] if "platt_scaling" in json_contents else 1
    intercept = json_contents["platt_intercept"] if "platt_intercept" in json_contents else 0
    intercept = np.log10(np.exp(intercept))

# load model
print("Loading model...")
division_model = load_division_model(_model_folder)

# create output folder
_output_folder = os.path.abspath(_output_folder)  # Convert to absolute path, as list_io changes the working directory
os.makedirs(_output_folder, exist_ok=True)

experiments_to_save = list()
for experiment_index, experiment in enumerate(list_io.load_experiment_list_file(_dataset_file)):
    # Check if output file exists already (in which case we skip this experiment)
    output_file = os.path.join(_output_folder, f"{experiment_index + 1}. {experiment.name.get_save_name()}."
                               + io.FILE_EXTENSION)
    if os.path.isfile(output_file):
        experiment.last_save_file = output_file
        experiments_to_save.append(experiment)
        print(f"Experiment {experiment_index + 1} ({experiment.name.get_save_name()}) already has divisions saved at"
              f" {output_file}. Skipping.")
        continue

    print(f"Working on experiment {experiment_index + 1}: {experiment.name}")
    division_model.predict_divisions(experiment, batch_size=_batch_size, image_channels=_images_channels,
                                     scale_factors_zyx=(_scale_factor_z, _scale_factor_xy, _scale_factor_xy),
                                     intensity_quantiles=(_intensity_quantile_min, _intensity_quantile_max))

    remove_division_oversegmentation(experiment, min_distance_dividing_um=_min_distance_dividing)

    print("Saving file...")
    io.save_data_to_json(experiment, output_file)
    experiments_to_save.append(experiment)


list_io.save_experiment_list_file(experiments_to_save,
                                  os.path.join(_output_folder, "_All" + list_io.FILES_LIST_EXTENSION))
print("Done!")