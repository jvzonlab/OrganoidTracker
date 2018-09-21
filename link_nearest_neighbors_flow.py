# Simple nearest-neighbor linking: every cell is linked to the nearest cell in the previous image. Physical and
# biological constraints are not taken into account.
from matplotlib import pyplot

from autotrack.core import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.linking import linker_for_experiment
from autotrack.linking import link_fixer
from autotrack.linking_analysis import comparison

# PARAMETERS
from autotrack.visualizer import image_visualizer

_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Data/" + _name + "/Automatic analysis/Positions/Manual.json"
_output_file = "../Data/" + _name + "/Automatic analysis/Links/Nearest neighbor.json"
_baseline_file = "../Data/" + _name + "/Automatic analysis/Links/Manual.json"
_images_folder = "../Data/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_flow_detection_radius = 50  # Radius of square in which the flow of neighboring cells is analyzed
_min_time_point = 0
_max_time_point = 5000  # Organoid moved position here
_flow_cycles = 1  # Amount of times linker_for_experiment.nearest_neighbor_using_flow is called
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions...")
io.load_positions_and_shapes_from_json(experiment, _positions_file)
print("Disovering images")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format, min_time_point=_min_time_point,
                                   max_time_point=_max_time_point)
print("Starting link process")
results_nearest_neighbor = linker_for_experiment.nearest_neighbor(experiment, tolerance=2,
                                                                  min_time_point=_min_time_point,
                                                                  max_time_point=_max_time_point)
results = results_nearest_neighbor
for i in range(_flow_cycles):
    results = linker_for_experiment.nearest_neighbor_using_flow(experiment, results, min_time_point=_min_time_point,
                                                                max_time_point=_max_time_point,
                                                                flow_detection_radius=_flow_detection_radius)
results = link_fixer.with_only_the_preferred_edges(results)
results_nearest_neighbor = link_fixer.with_only_the_preferred_edges(results_nearest_neighbor)
print("Writing results to file")
io.save_links_to_json(results, _output_file)
print("Visualizing")
experiment.particle_links(io.load_links_from_json(_baseline_file))
#experiment.particle_links(results_nearest_neighbor)
experiment.particle_links_scratch(results)
comparison.print_differences(experiment)
image_visualizer.show(experiment)
print("Done")
pyplot.show()

