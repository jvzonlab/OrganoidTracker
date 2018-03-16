# Compares two linking results. The baseline links are assumed to be 100% correct, any deviations from that are
# counted as errors. Solid lines in the figures represent the correct linking result, dotted lines any deviations from
# that.

from imaging import io, tifffolder, image_visualizer, Experiment
from linking_analysis import comparison
from matplotlib import pyplot

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Data/" + _name + "/Automatic analysis/Positions/Manual.json"
_images_folder = "../Data/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_time_point = 0
_max_time_point = 5000

_automatic_links = io.load_links_from_json("../Data/" + _name + "/Automatic analysis/Links/Smart nearest neighbor.json")
_baseline_links = io.load_links_from_json("../Data/" + _name + "/Automatic analysis/Links/Manual.json")
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
io.load_positions_from_json(experiment, _positions_file)
experiment.particle_links_scratch(_automatic_links)
experiment.particle_links(_baseline_links)

comparison.print_differences(_automatic_links, _baseline_links)

tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
vis = image_visualizer.show(experiment)

print("Done!")
pyplot.show()
