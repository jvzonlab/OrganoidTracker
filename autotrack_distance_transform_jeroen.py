#!/usr/bin/env python3
"""This script was based off an idea by Jeroen. Take the background, and create a distance transform to see how far each
background pixel is removed from the cells. This results in a kind of medial axis transform.

The distance from the cells is defined as either the distance to the threshold, or the distance to the cell point
position. For both definitions, images are written to disk.

Parameters:

- threshold_block_size: length and/or height of the 2d square used for the adaptive threshold (in pixels)
- max_distance_um: the distance transform is capped at this value
"""
from autotrack.imaging import tifffolder, io
from autotrack.config import ConfigFile
from autotrack.particle_detection import distance_transformer_for_experiment

# PARAMETERS
from autotrack.core.image_loader import ImageResolution
from autotrack.core.experiment import Experiment

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("distance_transform_jeroen")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_pixel_size_x_um = float(config.get_or_default("pixel_size_x_um", str(0.32), store_in_defaults=True))
_pixel_size_y_um = float(config.get_or_default("pixel_size_y_um", str(0.32), store_in_defaults=True))
_pixel_size_z_um = float(config.get_or_default("pixel_size_z_um", str(2), store_in_defaults=True))
_time_point_duration_m = float(config.get_or_default("time_point_duration_m", str(12), store_in_defaults=True))
_positions_input_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")

_threshold_block_size = int(config.get_or_default("threshold_block_size", str(51)))
_max_distance_um = float(config.get_or_default("max_distance_um", str(30)))
_output_folder = config.get_or_default("output_folder", "Automatic analysis/Distance transformed images")
config.save_and_exit_if_changed()
# END OF PARAMETERS

experiment = Experiment()
print("Loading cell positions...")
io.load_positions_and_shapes_from_json(experiment, _positions_input_file, min_time_point=_min_time_point,
                                       max_time_point=_max_time_point)
print("Discovering images...")
resolution = ImageResolution(_pixel_size_x_um, _pixel_size_y_um, _pixel_size_z_um, _time_point_duration_m)
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
experiment.image_resolution(resolution)
print("Running detection...")
distance_transformer_for_experiment.perform_for_experiment(experiment, _output_folder, _threshold_block_size,
                                                           _max_distance_um)
print("Done!")




