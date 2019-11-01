#!/usr/bin/env python3

"""Creates links between known nucleus positions at different time points. It always assigns nuclei to the nearsest
nucleus in the previous time point, which is a simplistic way of linking. However, if your time resolution is high
enough, it might actually work well."""

from ai_track.config import ConfigFile, config_type_int
from ai_track.imaging import io
from ai_track.image_loading import general_image_loader
from ai_track.linking import nearest_neighbor_linker, dpct_linker, cell_division_finder
from ai_track.linking.rational_scoring_system import RationalScoringSystem
from ai_track.linking_analysis import cell_error_finder, links_postprocessor

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("create_nearest_neighbor_links")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic positions.aut")
_links_output_file = config.get_or_default("output_file", "Automatic nearest neighbor links.aut")
config.save_and_exit_if_changed()
# END OF PARAMETERS


print("Loading cell positions and shapes...", _positions_file)
experiment = io.load_data_file(_positions_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Discovering images...")
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Performing nearest-neighbor linking...")
link_result = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=1, back=True, forward=False)
experiment.links = link_result
print("Checking results for common errors...")
warning_count = cell_error_finder.find_errors_in_experiment(experiment)
print("Writing results to file...")
io.save_data_to_json(experiment, _links_output_file)
print(f"Done! Found {warning_count} potential errors in the data.")
