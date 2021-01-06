#!/usr/bin/env python3

"""Creates links between known nucleus positions at different time points. Nucleus shape information (as obtained by
a Gaussian fit) is necessary for this."""

from organoid_tracker.config import ConfigFile, config_type_int
from organoid_tracker.imaging import io
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.linking import nearest_neighbor_linker, dpct_linker, cell_division_finder
from organoid_tracker.linking.rational_scoring_system import RationalScoringSystem
from organoid_tracker.linking_analysis import cell_error_finder, links_postprocessor

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("create_links")
_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                      " path here. Else, if you have a LIF file, please paste the path to that file"
                                      " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                      " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Gaussian fitted positions.aut")
_margin_xy = int(config.get_or_default("margin_xy", str(25)))
_link_weight = config.get_or_default("weight_links", str(20), comment="Penalty for link distance. Make this value"
                                      " higher if you're getting too many long-distance links. Lower this value if"
                                      " you're not ge tting enough links.", type=config_type_int)
_detection_weight = config.get_or_default("weight_detections", str(150), comment="Penalty for ignoring a detection."
                                           " Make this value higher if too many cells do not get any links.",
                                          type=config_type_int)
_division_weight = config.get_or_default("weight_division", str(30), comment="Score for creating a division. The"
                                         " higher, the more cell divisions will be created (although the volume of the"
                                         " cells still needs to be OK before any division is considered at all..",
                                         type=config_type_int)
_appearance_weight = config.get_or_default("weight_appearance", str(150), comment="Penalty for starting a track out of"
                                           " nowhere.", type=config_type_int)
_dissappearance_weight = config.get_or_default("weight_dissappearance", str(100), comment="Penalty for ending a track.",
                                               type=config_type_int)
_links_output_file = config.get_or_default("output_file", "Automatic links.aut")
config.save()
# END OF PARAMETERS


print("Loading cell positions and shapes...", _positions_file)
experiment = io.load_data_file(_positions_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Discovering images...")
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Performing nearest-neighbor linking...")
possible_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
print("Calculating scores of possible mothers...")
if experiment.scores.has_family_scores():
    print("    found existing scores, using those instead")
    scores = experiment.scores
elif experiment.position_data.has_position_data_with_name("shape"):
    score_system = RationalScoringSystem()
    scores = cell_division_finder.calculates_scores(experiment.images, experiment.position_data, possible_links, score_system)
else:
    print("    no scores found, no Gaussian fit found: cell divisions will NOT be recognized")
    scores = experiment.scores
print("Deciding on what links to use...")
link_result = dpct_linker.run(experiment.positions, experiment.position_data, possible_links, scores,
                              experiment.images.resolution(), link_weight=_link_weight,
                              detection_weight=_detection_weight, division_weight=_division_weight,
                              appearance_weight=_appearance_weight, dissappearance_weight=_dissappearance_weight)
print("Applying final touches...")
experiment.links = link_result
experiment.scores = scores
links_postprocessor.postprocess(experiment, margin_xy=_margin_xy)
print("Checking results for common errors...")
warning_count, no_links_count = cell_error_finder.find_errors_in_experiment(experiment)
print("Writing results to file...")
io.save_data_to_json(experiment, _links_output_file)
print(f"Done! Found {warning_count} potential errors in the data. In addition, {no_links_count} positions didn't get"
      f" links.")
