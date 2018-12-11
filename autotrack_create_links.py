#!/usr/bin/env python3

from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.core.links import ParticleLinks
from autotrack.core.resolution import ImageResolution
from autotrack.imaging import tifffolder, io
from autotrack.linking import nearest_neighbor_linker, dpct_linker, cell_division_finder
from autotrack.linking.rational_scoring_system import RationalScoringSystem
from autotrack.linking_analysis import cell_error_finder, links_postprocessor

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("create_links")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_pixel_size_x_um = float(config.get_or_default("pixel_size_x_um", str(0.32), store_in_defaults=True))
_pixel_size_y_um = float(config.get_or_default("pixel_size_y_um", str(0.32), store_in_defaults=True))
_pixel_size_z_um = float(config.get_or_default("pixel_size_z_um", str(2), store_in_defaults=True))
_time_point_duration_m = float(config.get_or_default("time_point_duration_m", str(12), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Gaussian fit.json")
_margin_xy = int(config.get_or_default("margin_xy", str(50)))
_links_output_file = config.get_or_default("output_file", "Automatic analysis/Links/Smart nearest neighbor.aut")
config.save_and_exit_if_changed()
# END OF PARAMETERS


experiment = Experiment()
resolution = ImageResolution(_pixel_size_x_um, _pixel_size_y_um, _pixel_size_z_um, _time_point_duration_m)
experiment.image_resolution(resolution)
print("Loading cell positions and shapes...", _positions_file)
io.load_positions_and_shapes_from_json(experiment, _positions_file, min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Discovering images...")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Performing nearest-neighbor linking...")
possible_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)
print("Calculating scores of possible mothers...")
score_system = RationalScoringSystem()
scores = cell_division_finder.calculates_scores(experiment.image_loader(), experiment.particles, possible_links, score_system)
print("Deciding on what links to use...")
link_result = dpct_linker.run(experiment.particles, possible_links, scores)
print("Applying final touches...")
experiment.links = link_result
experiment.scores = scores
links_postprocessor.postprocess(experiment, margin_xy=_margin_xy)
print("Checking results for common errors...")
cell_error_finder.apply(experiment)
print("Writing results to file...")
io.save_data_to_json(experiment, _links_output_file)

print("Done!")
