# Small visualization tool. Keeps only one TIFF file in memory at a time, so it should start up fast and have a
# reasonable low memory footprint.

from imaging import io, tifffolder, image_visualizer, Experiment
from matplotlib import pyplot
from os import path

# PARAMETERS
# _positions_file and _links_file are optional: if they point to a file that does not exist, simply no positions or
# links data will be displayed.
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format)
image_visualizer.show(experiment)

print("Done!")
pyplot.show()
