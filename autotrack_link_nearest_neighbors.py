#!/usr/bin/env python3

"""Simple nearest-neighbor linking: every cell is linked to the nearest cell in the previous image. Physical and
biological constraints are not taken into account. The visualizer shows not only the nearest neighbors, but also other
possible links: those alternative links are shown as dotted lines.
"""
from autotrack import gui
from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.linking import linker_for_experiment, link_util
from autotrack.visualizer import image_visualizer

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("link_nearest_neighbors")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_output_file = config.get_or_default("links_file", "Automatic analysis/Links/Nearest neighbor.json")
config.save_and_exit_if_changed()
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions...")
io.load_positions_and_shapes_from_json(experiment, _positions_file, min_time_point=_min_time_point,
                                       max_time_point=_max_time_point)
print("Discovering images...")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format, min_time_point=_min_time_point,
                                   max_time_point=_max_time_point)
print("Starting link process...")
results_all = linker_for_experiment.nearest_neighbor(experiment, tolerance=2)
results = link_util.with_only_the_preferred_edges(results_all)
print("Writing results to file...")
io.save_links_to_json(results, _output_file)
print("Visualizing...")
experiment.particle_links_scratch(results_all)
experiment.particle_links(results)
image_visualizer.show(experiment)
print("Done!")
gui.mainloop()
