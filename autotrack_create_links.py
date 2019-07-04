#!/usr/bin/env python3

"""Creates links between known nucleus positions at different time points. Nucleus shape information (as obtained by
a Gaussian fit) is necessary for this."""

from autotrack.config import ConfigFile
from autotrack.imaging import io
from autotrack.image_loading import general_image_loader
from autotrack.linking import nearest_neighbor_linker, dpct_linker, cell_division_finder
from autotrack.linking.rational_scoring_system import RationalScoringSystem
from autotrack.linking_analysis import cell_error_finder, links_postprocessor

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
_margin_xy = int(config.get_or_default("margin_xy", str(50)))
_links_output_file = config.get_or_default("output_file", "Automatic links.aut")
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
scores = cell_division_finder.calculates_scores(experiment.images, experiment.positions, possible_links, score_system)
print("Deciding on what links to use...")
link_result = dpct_linker.run(experiment.positions, possible_links, scores, experiment.images.resolution())
print("Applying final touches...")
experiment.links = link_result
experiment.scores = scores
links_postprocessor.postprocess(experiment, margin_xy=_margin_xy)
print("Checking results for common errors...")
cell_error_finder.apply(experiment)
print("Writing results to file...")
io.save_data_to_json(experiment, _links_output_file)

print("Done!")
