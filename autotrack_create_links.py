#!/usr/bin/env python3

from autotrack.config import ConfigFile
from autotrack.core import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.linking import linker_for_experiment
from autotrack.linking import link_fixer_casebycase
from autotrack.linking.rational_scoring_system import RationalScoringSystem

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
_mitotic_radius = int(config.get_or_default("mitotic_radius", str(3)))
_flow_detection_radius = int(config.get_or_default("flow_detection_radius", str(50)))
config.save_if_changed()
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions and shapes...", _positions_file)
io.load_positions_and_shapes_from_json(experiment, _positions_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Discovering images...")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Performing nearest-neighbor linking...")
possible_links = linker_for_experiment.nearest_neighbor(experiment, tolerance=2)
print("Improving links by analyzing local average movement...")
possible_links = linker_for_experiment.nearest_neighbor_using_flow(experiment, possible_links,
                                                                   flow_detection_radius=_flow_detection_radius)
print("Deciding on what links to use...")
score_system = RationalScoringSystem(_mitotic_radius)
link_result = link_fixer_casebycase.prune_links(experiment, possible_links, score_system)
print("Writing results to file...")
io.save_links_and_scores_to_json(experiment, link_result, _links_output_file)
print("Done")
