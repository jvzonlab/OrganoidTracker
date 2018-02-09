# Compares two linking results. The baseline links are assumed to be 100% correct, any deviations from that are
# counted as errors. Solid lines in the figures represent the correct linking result, dotted lines any deviations from
# that.

from imaging import io, tifffolder, visualizer, Experiment
from matplotlib import pyplot
from manual_tracking import links_extractor
import networkx
from os import path

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Manual positions.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"

_automatic_links = io.load_links_from_json("../Results/" + _name + "/Nearest neighbor links.json")
_baseline_links = links_extractor.extract_from_tracks("../Results/" + _name + "/Manual tracks")
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
io.load_positions_from_json(experiment, _positions_file)
experiment.particle_links(_automatic_links)
experiment.particle_links_baseline(_baseline_links)

tifffolder.load_images_from_folder(experiment, _images_folder, _images_format)
vis = visualizer.visualize(experiment)

print("Done!")
pyplot.show()
