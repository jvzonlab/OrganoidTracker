# Simple nearest-neighbor linking: every cell is linked to the nearest cell in the previous image. Physical and
# biological constraints are not taken into account.
from matplotlib import pyplot

from autotrack.core import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.linking import linker_for_experiment

# PARAMETERS
from autotrack.visualizer import image_visualizer

_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Data/" + _name + "/Automati analysis/Positions/Manual.json"
_output_file = "../Data/" + _name + "/Automatic analysis/Links/Nearest neighbor.json"
_images_folder = "../Data/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_time_point = 0
_max_time_point = 115  # Organoid moved position here
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions...")
io.load_positions_and_shapes_from_json(experiment, _positions_file)
print("Disovering images")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format, min_time_point=_min_time_point,
                                   max_time_point=_max_time_point)
print("Staring link process")
results = linker_for_experiment.nearest_neighbor(experiment,
                                                 min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Writing results to file")
io.save_links_to_json(results, _output_file)
print("Visualizing")
image_visualizer.show(experiment)
print("Done")
pyplot.show()

