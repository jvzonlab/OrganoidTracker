#!/usr/bin/env python3

"""Little bit more complex linking than just nearest-neighbor linking. Physical and biological constraints are still not
taken into account. Based on the average movement of the cells (as determined by nearest, cells can be linked to a cell
that is not the nearest neighbor.
"""
from autotrack import gui
from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.linking import linker_for_experiment
from autotrack.linking import link_util
from autotrack.visualizer import image_visualizer

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("link_nearest_neighbors_flow")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_output_file = config.get_or_default("links_file", "Automatic analysis/Links/Flow-corrected nearest neighbor.json")
# "Radius" of square in which the flow of neighboring cells is analyzed
_flow_detection_radius = int(config.get_or_default("flow_detection_radius", str(50)))
# Amount of times linker_for_experiment.nearest_neighbor_using_flow is called
_flow_cycles = int(config.get_or_default("flow_cycles", str(1)))
config.save_and_exit_if_changed()
# END OF PARAMETERS

experiment = Experiment()
print("Loading cell positions...")
io.load_positions_and_shapes_from_json(experiment, _positions_file, min_time_point=_min_time_point,
                                       max_time_point=_max_time_point)
print("Discovering images...")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format, min_time_point=_min_time_point,
                                   max_time_point=_max_time_point)
print("Performing nearest neighbor linking...")
results_nearest_neighbor = linker_for_experiment.nearest_neighbor(experiment, tolerance=2)
results = results_nearest_neighbor
for i in range(_flow_cycles):
    print("Preforming average flow-based corrections, iteration " + str(i) + "...")
    results = linker_for_experiment.nearest_neighbor_using_flow(experiment, results,
                                                                flow_detection_radius=_flow_detection_radius)
results = link_util.with_only_the_preferred_edges(results)
results_nearest_neighbor = link_util.with_only_the_preferred_edges(results_nearest_neighbor)
print("Writing results to file...")
io.save_links_to_json(results, _output_file)
print("Visualizing..")
experiment.particle_links_scratch(results)
image_visualizer.show(experiment)
print("Done!")
gui.mainloop()

