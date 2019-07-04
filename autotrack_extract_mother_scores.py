#!/usr/bin/env python3

"""Script for testing the mother scoring system. Output a CSV file with the scores of all putative mothers, based on
simple nearest-neighbor linking. The data is compared with data of the actual mothers.
"""
from autotrack.config import ConfigFile
from autotrack.imaging import io
from autotrack.image_loading import general_image_loader
from autotrack.linking import nearest_neighbor_linker, cell_division_finder
from autotrack.linking.rational_scoring_system import RationalScoringSystem
from autotrack.linking_analysis import scores_dataframe

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("extract_mother_scores")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_baseline_links_file = config.get_or_default("baseline_links_file", "Manual links.aut")
_output_file = config.get_or_default("output_csv_file", "Mother scores.csv")
config.save_and_exit_if_changed()

# END OF PARAMETERS

print("Starting...")
experiment = io.load_data_file(_baseline_links_file,
                                min_time_point=_min_time_point, max_time_point=_max_time_point)
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)

print("Discovering possible links using greedy nearest-neighbor...")
nearest_neighbor_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)

print("Scoring all possible mothers")
scoring_system = RationalScoringSystem()
real_mothers = set(cell_division_finder.find_mothers(experiment.links))
putative_families = cell_division_finder.find_families(nearest_neighbor_links, warn_on_many_daughters=False)
dataframe = scores_dataframe.create(experiment, putative_families, scoring_system, real_mothers)
io.save_dataframe_to_csv(dataframe, _output_file)

print("Done!")
