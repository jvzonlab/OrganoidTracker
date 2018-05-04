"""Script for testing the mother scoring system. Output a CSV file with the scores of all putative mothers, based on
simple nearest-neighbor linking. The data is compared with data of the actual mothers.
"""
from config import ConfigFile
from imaging import io, tifffolder
from core import Experiment
from linking import linker_for_experiment, mother_finder
from linking.rational_scoring_system import RationalScoringSystem
from linking_analysis import scores_dataframe

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
_mitotic_radius = int(config.get_or_default("mitotic_radius", str(3)))
_shape_detection_radius = int(config.get_or_default("shape_detection_radius", str(16)))
config.save_if_changed()
_baseline_links = io.load_links_from_json(_baseline_links_file)

# END OF PARAMETERS

print("Starting...")
experiment = Experiment()
io.load_positions_and_shapes_from_json(experiment, _positions_file)
experiment.particle_links(_baseline_links)
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)

print("Discovering possible links using greedy nearest-neighbor...")
possible_links = linker_for_experiment.nearest_neighbor(experiment, tolerance=2,
                                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
experiment.particle_links_scratch(possible_links)

print("Scoring all possible mothers")
scoring_system = RationalScoringSystem(_mitotic_radius)
real_families = set(mother_finder.find_families(_baseline_links, warn_on_many_daughters=True))
putative_families = mother_finder.find_families(possible_links, warn_on_many_daughters=False)
dataframe = scores_dataframe.create(experiment, putative_families, scoring_system, real_families)
io.save_dataframe_to_csv(dataframe, _output_file)

print("Done!")
