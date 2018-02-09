# Small visualization tool. Keeps only one TIFF file in memory at a time, so it should start up fast and have a
# reasonable low memory footprint.

from imaging import io, tifffolder, image_visualizer, Experiment
from matplotlib import pyplot
from os import path

# PARAMETERS
# _positions_file and _links_file are optional: if they point to a file that does not exist, simply no positions or
# links data will be displayed.
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Manual positions.json"
_links_file = "../Results/" + _name + "/Manual links.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
if path.exists(_positions_file):
    io.load_positions_from_json(experiment, _positions_file)
if path.exists(_links_file):
    experiment.particle_links_baseline(io.load_links_from_json( _links_file))
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format)
image_visualizer.show(experiment)

print("Done!")
pyplot.show()
