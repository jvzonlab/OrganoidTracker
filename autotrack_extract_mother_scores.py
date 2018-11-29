#!/usr/bin/env python3

"""Script for testing the mother scoring system. Output a CSV file with the scores of all putative mothers, based on
simple nearest-neighbor linking. The data is compared with data of the actual mothers.
"""
from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.linking import linker_for_experiment, mother_finder
from autotrack.linking.rational_scoring_system import RationalScoringSystem
from autotrack.linking_analysis import scores_dataframe

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("extract_mother_scores")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_baseline_links_file = config.get_or_default("baseline_links_file", "Automatic analysis/Links/Manual.json")
_output_file = config.get_or_default("ouput_csv_file", "Automatic analysis/Links/Scoring analysis.csv")
config.save_and_exit_if_changed()

# END OF PARAMETERS

print("Starting...")
experiment = Experiment()
io.load_positions_and_shapes_from_json(experiment, _positions_file,
                                       min_time_point=_min_time_point, max_time_point=_max_time_point)
_baseline_links = io.load_links_from_json(_baseline_links_file,
                                          min_time_point=_min_time_point, max_time_point=_max_time_point)
experiment.links = _baseline_links
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)

print("Discovering possible links using greedy nearest-neighbor...")
possible_links = linker_for_experiment.nearest_neighbor(experiment, tolerance=2)
experiment.links = possible_links

print("Scoring all possible mothers")
scoring_system = RationalScoringSystem()
real_mothers = set(mother_finder.find_mothers(_baseline_links))
putative_families = mother_finder.find_families(possible_links, warn_on_many_daughters=False)
dataframe = scores_dataframe.create(experiment, putative_families, scoring_system, real_mothers)
io.save_dataframe_to_csv(dataframe, _output_file)

print("Done!")
