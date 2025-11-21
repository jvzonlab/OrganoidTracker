"""Predicts cell positions using an already-trained convolutional neural network."""
import _keras_environment
from organoid_tracker.core.image_loader import ImageChannel

_keras_environment.activate()

import os

from organoid_tracker.config import ConfigFile, config_type_int
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.imaging import io, list_io
from organoid_tracker.neural_network.position_detection_cnn.position_predictor import load_position_model

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("predict_positions")
_dataset_file = config.get_or_prompt("dataset_file", "Please paste the path here to the dataset file."
                                     " You can generate such a file from OrganoidTracker using File -> Tabs -> "
                                     " all tabs.", store_in_defaults=True)

if _dataset_file != '':
    experiment_count = list_io.count_experiments_in_list_file(_dataset_file)
    experiment_list = list_io.load_experiment_list_file(_dataset_file)
else:
    # if not _dataset_file is defined, we look in the defaults for an images folder to construct a single experiment
    _images_folder = config.get_or_prompt("images_container",
                                          "If you have a folder of image files, please paste the folder"
                                          " path here. Else, if you have a LIF file, please paste the path to that file"
                                          " here.", store_in_defaults=True)
    _images_format = config.get_or_prompt("images_pattern",
                                          "What are the image file names? (Use {time:03} for three digits"
                                          " representing the time point, use {channel} for the channel)",
                                          store_in_defaults=True)

    _output_file = config.get_or_default("positions_output_file", "Automatic positions.aut",
                                         comment="Output file for the positions, can be viewed using the visualizer program.")

    _min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
    _max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))

    experiment_list = Experiment()
    general_image_loader.load_images(experiment_list, _images_folder, _images_format,
                                     min_time_point=_min_time_point, max_time_point=_max_time_point)

    experiment_count = 1
    experiment_list = [experiment_list]

_patch_shape_unbuffered_y = config.get_or_default("patch_shape_y", str(320), type=config_type_int, comment="Maximum patch size to use for predictions."
                                       " Make this smaller if you run out of video card memory.")
_patch_shape_unbuffered_x = config.get_or_default("patch_shape_x", str(320), type=config_type_int)

_buffer_z = config.get_or_default("buffer_z", str(1), comment="Buffer space to use when stitching multiple patches"
                                                              " together. Added on top of the patch shape.", type=config_type_int)
_buffer_y = config.get_or_default("buffer_y", str(32), type=config_type_int, comment="buffer_y * 2 + patch_shape_y needs to be a multiple of 32")
_buffer_x = config.get_or_default("buffer_x", str(32), type=config_type_int, comment="Same for buffer_x")

_model_folder = config.get_or_prompt("model_folder", "Please paste the path here to the \"trained_model\" folder containing the trained model.")
_output_folder = config.get_or_default("positions_output_folder", "Automatic positions", comment="Output folder for the positions, can be viewed using the visualizer program.")
_channels_str = config.get_or_default("images_channels", str(1), comment="Index(es) of the channels to use. Use \"3\" to use the third channel for predictions. Use \"1,3,4\" to use the sum of the first, third and fourth channel for predictions.")
_images_channels = {ImageChannel(index_one=int(part)) for part in _channels_str.split(",")}
_peak_min_distance_px = int(config.get_or_default("peak_min_distance_px", str(6), comment="Minimum distance in pixels"
                                                                                          " between detected positions."))
_threshold = float(config.get_or_default("threshold", str(0.1), comment="Minimum peak intensity."))

_debug_folder = config.get_or_default("predictions_output_folder", "",
                                      comment="If you want to see the raw prediction images, paste the path to a folder here. In that folder, a prediction image will be placed for each time point.")
if len(_debug_folder) == 0:
    _debug_folder = None
config.save()
# END OF PARAMETERS

# Load models
print("Loading model...")
model = load_position_model(_model_folder)

# Make folders
if _debug_folder is not None:
    os.makedirs(_debug_folder, exist_ok=True)

if _dataset_file != '':
    os.makedirs(_output_folder, exist_ok=True)

# Loop through experiments
experiments_to_save = list()
for experiment_index, experiment in enumerate(experiment_list):
    print(f"\nWorking on experiment {experiment_index + 1}/{experiment_count}: {experiment.name}...")
    # Check if output file exists already (in which case we skip this experiment)
    if _dataset_file != '':
        output_file = os.path.join(_output_folder, f"{experiment_index + 1}. {experiment.name.get_save_name()}."
                                   + io.FILE_EXTENSION)
    else:
        output_file = _output_file

    # Clear any existing tracking data
    experiment.clear_tracking_data()

    # Load existing output file - useful when the script was stopped halfway previously
    if os.path.isfile(output_file):
        print(f"Output file {output_file} already exists, continuing were we left off.")
        io.load_data_file(output_file, experiment=experiment)

    debug_folder_experiment = os.path.join(_debug_folder, f"{experiment_index + 1}. {experiment.name.get_save_name()}") \
        if _debug_folder is not None else None
    model.predict(experiment, debug_folder_experiment=debug_folder_experiment, image_channels=_images_channels,
                  patch_shape_unbuffered_yx=(_patch_shape_unbuffered_y, _patch_shape_unbuffered_x),
                  peak_min_distance_px=_peak_min_distance_px,
                  buffer_size_zyx=(_buffer_z, _buffer_y, _buffer_x),
                  threshold=_threshold,
                  output_file=output_file)

    if _dataset_file != '':
        # Collect for writing AUTLIST file
        experiments_to_save.append(experiment)

if _dataset_file != '':
    list_io.save_experiment_list_file(experiments_to_save,
                                  os.path.join(_output_folder, "_All" + list_io.FILES_LIST_EXTENSION))


print("Done!")
