#!/usr/bin/env python3

from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.linking import linker_for_experiment, dpct_linking, mother_finder
from autotrack.linking.cheating_score_system import CheatingScoringSystem
from autotrack.linking.rational_scoring_system import RationalScoringSystem
from autotrack.linking_analysis import logical_tests

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("create_links")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_links_output_file = config.get_or_default("links_output_file", "Automatic analysis/Links/Smart nearest neighbor.json")
config.save_and_exit_if_changed()
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions and shapes...", _positions_file)
io.load_positions_and_shapes_from_json(experiment, _positions_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Discovering images...")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Performing nearest-neighbor linking...")
possible_links = linker_for_experiment.nearest_neighbor(experiment, tolerance=2, over_previous=False)
print("Calculating scores of possible mothers...")
score_system = RationalScoringSystem()
scores = mother_finder.calculates_scores(experiment.image_loader(), experiment.particles, possible_links, score_system)
print("Deciding on what links to use...")
link_result = dpct_linking.run(experiment.particles, possible_links, scores)
print("Checking results for common errors...")
logical_tests.apply(scores, experiment.particles, link_result)
print("Writing results to file...")
io.save_links_and_scores_to_json(link_result, scores, _links_output_file)

print("Done!")
