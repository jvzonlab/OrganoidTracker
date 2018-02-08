# Small visualization tool
# Note: you can also use track_manually to compare

from imaging import io, tifffolder, visualizer, Experiment
from matplotlib import pyplot
from os import path

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Manual positions.json"
_links_file = "../Results/" + _name + "/Nearest neighbor links.json "
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
if path.exists(_positions_file):
    io.load_positions_from_json(experiment, _positions_file)
if path.exists(_links_file):
    io.load_links_from_json(experiment, _links_file)
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format)
vis = visualizer.visualize(experiment)

print("Done!")
pyplot.show()
