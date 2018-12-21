#!/usr/bin/env python3

"""Compares two sets of positions. Used to calculate the recall and precision."""
from autotrack.comparison import positions_comparison
from autotrack.config import ConfigFile
from autotrack.core.resolution import ImageResolution
from autotrack.imaging import io

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("compare_positions")
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_pixel_size_x_um = float(config.get_or_default("pixel_size_x_um", str(0.32), store_in_defaults=True))
_pixel_size_y_um = float(config.get_or_default("pixel_size_y_um", str(0.32), store_in_defaults=True))
_pixel_size_z_um = float(config.get_or_default("pixel_size_z_um", str(2), store_in_defaults=True))
_time_point_duration_m = float(config.get_or_default("time_point_duration_m", str(12), store_in_defaults=True))
_ground_truth_file = config.get_or_prompt("positions_ground_truth_file", "In what file are the positions of the ground truth stored?")
_automatic_file = config.get_or_prompt("positions_automatic_file", "In what file are the positions of the experiment stored?")
_max_distance_um = float(config.get_or_default("max_distance_um", str(5)))
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Starting...")
ground_truth = io.load_data_file(_ground_truth_file, _min_time_point, _max_time_point)
ground_truth.images.set_resolution(ImageResolution(_pixel_size_x_um, _pixel_size_y_um, _pixel_size_z_um, _time_point_duration_m))
automatic_data = io.load_data_file(_automatic_file, _min_time_point, _max_time_point)

print("Comparing...")
result = positions_comparison.compare_positions(ground_truth, automatic_data, max_distance_um=_max_distance_um)
print(result)
result.calculate_detection_statistics().debug_plot()

print("Done!")
