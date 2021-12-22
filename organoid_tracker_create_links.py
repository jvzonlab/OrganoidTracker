#!/usr/bin/env python3

"""Creates links between known nucleus positions at different time points. Nucleus shape information (as obtained by
a Gaussian fit) is necessary for this."""

from organoid_tracker.config import ConfigFile, config_type_int, config_type_float
from organoid_tracker.imaging import io
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.linking import dpct_linker
from organoid_tracker.linking.dpct_linker import calculate_appearance_penalty
from organoid_tracker.linking_analysis import cell_error_finder

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
_positions_file = config.get_or_default("positions_file", "positions.aut")
_margin_um = int(config.get_or_default("margin_um", str(8)))
_link_weight = config.get_or_default("weight_links", str(1), comment="Penalty for link distance. Make this value"
                                      " higher if you're getting too many long-distance links. Lower this value if"
                                      " you're not ge tting enough links.", type=config_type_int)
_detection_weight = config.get_or_default("weight_detections", str(1), comment="Penalty for ignoring a detection."
                                           " Make this value higher if too many cells do not get any links.",
                                          type=config_type_int)
_division_weight = config.get_or_default("weight_division", str(1), comment="Score for creating a division. The"
                                         " higher, the more cell divisions will be created (although the volume of the"
                                         " cells still needs to be OK before any division is considered at all..",
                                         type=config_type_int)
_appearance_weight = config.get_or_default("weight_appearance", str(1), comment="Penalty for starting a track out of"
                                           " nowhere.", type=config_type_int)
_disappearance_weight = config.get_or_default("weight_dissappearance", str(1), comment="Penalty for ending a track.",
                                               type=config_type_int)
min_appearance_probability = config.get_or_default("min_appearance_probability", str(0.01), comment="Estimate of a track appearing (no division).",
                                               type=config_type_float)
min_disappearance_probability = config.get_or_default("min_disappearance_probability", str(0.01), comment="Estimate of a track disappearing (no division).",
                                               type=config_type_float)
_links_output_file = config.get_or_default("output_file", "Automatic links.aut")
config.save()

# END OF PARAMETERS
print(min_appearance_probability)

print("Loading cell positions and link/division predictions...", _positions_file)
experiment = io.load_data_file(_positions_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
possible_links = experiment.links

print("Discovering images...")
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)

print("calculate appearance and disappearance probabilities...")
experiment = calculate_appearance_penalty(experiment, min_appearance_probability=min_appearance_probability, name="appearance_penalty", buffer_distance=_margin_um)
experiment = calculate_appearance_penalty(experiment, min_appearance_probability=min_disappearance_probability, name="disappearance_penalty", buffer_distance=_margin_um)

print("Deciding on what links to use...")
link_result, naive_links = dpct_linker.run(experiment.positions, experiment.position_data, possible_links, link_data=experiment.link_data, link_weight=1,
                              detection_weight=_detection_weight, division_weight=_division_weight,
                              appearance_weight=_appearance_weight, dissappearance_weight=_disappearance_weight)

print("Applying final touches...")
experiment.links = link_result

print("Checking results for common errors...")
warning_count, no_links_count = cell_error_finder.find_errors_in_experiment(experiment)
print("Writing results to file...")
io.save_data_to_json(experiment, _links_output_file)

print(f"Done! Found {warning_count} potential errors in the data. In addition, {no_links_count} positions didn't get"
      f" links.")

# This includes the unpruned set of links, to help gain insight in how the dpct linker performed
print("Writing naive results to file...")
experiment.links = naive_links
io.save_data_to_json(experiment, 'naive' + _links_output_file)