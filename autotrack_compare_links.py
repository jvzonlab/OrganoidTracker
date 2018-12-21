#!/usr/bin/env python3

"""Compares two linking results. The baseline links are assumed to be 100% correct, any deviations from that are
counted as errors."""
from autotrack.comparison import links_comparison
from autotrack.config import ConfigFile
from autotrack.core.resolution import ImageResolution
from autotrack.imaging import io

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("compare_links")
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_pixel_size_x_um = float(config.get_or_default("pixel_size_x_um", str(0.32), store_in_defaults=True))
_pixel_size_y_um = float(config.get_or_default("pixel_size_y_um", str(0.32), store_in_defaults=True))
_pixel_size_z_um = float(config.get_or_default("pixel_size_z_um", str(2), store_in_defaults=True))
_time_point_duration_m = float(config.get_or_default("time_point_duration_m", str(12), store_in_defaults=True))

_automatic_links_file = config.get_or_default("automatic_links_file", "Automatic links.aut")
_baseline_links_file = config.get_or_default("baseline_links_file", "Manual links.aut")
_max_distance_um = float(config.get_or_default("max_distance_um", str(5)))
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Starting...")

scratch_experiment = io.load_data_file(_automatic_links_file, _min_time_point, _max_time_point)
baseline_experiment = io.load_data_file(_baseline_links_file, _min_time_point, _max_time_point)
baseline_experiment.image_resolution(ImageResolution(_pixel_size_x_um, _pixel_size_y_um, _pixel_size_z_um, _time_point_duration_m))

print("Comparing...")
report = links_comparison.compare_links(baseline_experiment, scratch_experiment, _max_distance_um)
print(report)

print("Done!")
