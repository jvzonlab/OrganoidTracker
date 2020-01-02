#!/usr/bin/env python3

"""Creates links between known nucleus positions at different time points. Nucleus shape information (as obtained by
a Gaussian fit) is necessary for this."""

from ai_track.config import ConfigFile, config_type_int, config_type_float
from ai_track.image_loading import general_image_loader
from ai_track.imaging import io
from ai_track.linking import nearest_neighbor_linker, dpct_linker
from ai_track.linking_analysis import cell_error_finder, linking_markers

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
                                      " you're not ge tting enough links.", type=config_type_float)
_detection_weight = config.get_or_default("weight_detections", str(150), comment="Penalty for ignoring a detection."
                                           " Make this value higher if too many cells do not get any links.",
                                          type=config_type_float)
_division_weight = config.get_or_default("weight_division", str(30), comment="Score for creating a division. The"
                                         " higher, the more cell divisions will be created (although the volume of the"
                                         " cells still needs to be OK before any division is considered at all..",
                                         type=config_type_float)
_appearance_weight = config.get_or_default("weight_appearance", str(150), comment="Penalty for starting a track out of"
                                           " nowhere.", type=config_type_float)
_dissappearance_weight = config.get_or_default("weight_dissappearance", str(100), comment="Penalty for ending a track.",
                                               type=config_type_float)
_links_output_file = config.get_or_default("output_file", "Automatic links.aut")
config.save()
# END OF PARAMETERS


print("Loading cell positions...", _positions_file)
experiment = io.load_data_file(_positions_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Discovering images...")
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
if linking_markers.has_mother_scores(experiment.position_data):
    print("Found existing mother scores.")
else:
    print("Warning! No mother score information found. This means that cell divisions are NOT detected.")
print("Performing nearest-neighbor linking...")
possible_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
print("Deciding on what links to use...")
link_result = dpct_linker.run(experiment.positions, possible_links, experiment.position_data,
                              experiment.images.resolution(), link_weight=_link_weight,
                              detection_weight=_detection_weight, division_weight=_division_weight,
                              appearance_weight=_appearance_weight, dissappearance_weight=_dissappearance_weight)
print("Applying final touches...")
experiment.links = link_result
#links_postprocessor.postprocess(experiment, margin_xy=_margin_xy)
print("Checking results for common errors...")
warning_count = cell_error_finder.find_errors_in_experiment(experiment)
print("Writing results to file...")
io.save_data_to_json(experiment, _links_output_file)
print(f"Done! Found {warning_count} potential errors in the data.")
