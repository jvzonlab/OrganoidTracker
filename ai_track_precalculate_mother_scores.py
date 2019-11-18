#!/usr/bin/env python3

"""Script for testing the mother scoring system. Output a CSV file with the scores of all putative mothers, based on
simple nearest-neighbor linking. The data is compared with data of the actual mothers.
"""
from ai_track.config import ConfigFile
from ai_track.imaging import io
from ai_track.image_loading import general_image_loader
from ai_track.linking import nearest_neighbor_linker, cell_division_finder
from ai_track.linking.rational_scoring_system import RationalScoringSystem
from ai_track.linking_analysis import scores_dataframe

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("precalculate_mother_scores")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("input_file", "Gaussian fitted positions.aut")
_output_file = config.get_or_default("output_file", "Scored positions.aut")
config.save_and_exit_if_changed()

# END OF PARAMETERS

print("Loading cell positions and shapes...", _positions_file)
experiment = io.load_data_file(_positions_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Discovering images...")
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Performing nearest-neighbor linking...")
possible_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
print("Calculating scores of possible mothers...")
score_system = RationalScoringSystem()
experiment.scores = cell_division_finder.calculates_scores(experiment.images, experiment.positions, possible_links, score_system)
print("Saving result..")
io.save_data_to_json(experiment, _output_file)
print("Done!")
