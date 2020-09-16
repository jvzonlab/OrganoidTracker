#!/usr/bin/env python3

"""Compares two linking results on the level of individual links. Therefore, this doesn't report missing cell divisions
and deaths, but it can give you a percentage of the number of links that were correct.

The baseline links are assumed to be 100% correct, any deviations from that are counted as errors."""
from organoid_tracker.comparison import links_comparison, report_json_io
from organoid_tracker.config import ConfigFile, config_type_json_file, config_type_int, config_type_float
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.imaging import io

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("compare_links")
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_pixel_size_x_um = float(config.get_or_default("pixel_size_x_um", str(0.32), store_in_defaults=True))
_pixel_size_y_um = float(config.get_or_default("pixel_size_y_um", str(0.32), store_in_defaults=True))
_pixel_size_z_um = float(config.get_or_default("pixel_size_z_um", str(2), store_in_defaults=True))
_time_point_duration_m = float(config.get_or_default("time_point_duration_m", str(12), store_in_defaults=True))

_automatic_links_file = config.get_or_prompt("automatic_links_file", "In what file are the new links stored?")
_baseline_links_file = config.get_or_prompt("baseline_links_file", "In what file are the original links stored?")
_max_distance_um = config.get_or_default("max_distance_um", str(5), type=config_type_float, comment="Maximum distance"
                                         " between positions in different data sets for them to be considered equal.")
_margin_xy_px = config.get_or_default("margin_xy_px", str(-1), type=config_type_int, comment="If you set this to 0 or"
                                      " higher, only positions in the images that far from the border will be"
                                      " considered.")
if _margin_xy_px >= 0:
    # Load images
    _images_folder = config.get_or_prompt("images_container",
                                          "If you have a folder of image files, please paste the folder"
                                          " path here. Else, if you have a LIF file, please paste the path to that file"
                                          " here.", store_in_defaults=True)
    _images_format = config.get_or_prompt("images_pattern",
                                          "What are the image file names? (Use {time:03} for three digits"
                                          " representing the time point, use {channel} for the channel)",
                                          store_in_defaults=True)
else:
    _images_folder = None
    _images_format = None
_output_file = config.get_or_default("output_file", "", type=config_type_json_file)
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Starting...")
scratch_experiment = io.load_data_file(_automatic_links_file, _min_time_point, _max_time_point)
baseline_experiment = io.load_data_file(_baseline_links_file, _min_time_point, _max_time_point)
baseline_experiment.images.set_resolution(ImageResolution(_pixel_size_x_um, _pixel_size_y_um, _pixel_size_z_um, _time_point_duration_m))

if _images_folder is not None:
    print("Discovering images...")
    general_image_loader.load_images(baseline_experiment, _images_folder, _images_format,
                                     min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Comparing...")
report = links_comparison.compare_links(baseline_experiment, scratch_experiment, _max_distance_um, _margin_xy_px)
print("Writing report...")
if _output_file:
    report_json_io.save_report(report, _output_file)
print(report)
report.calculate_time_correctness_statistics().debug_plot()
report.calculate_z_correctness_statistics().debug_plot()

print("Done!")
