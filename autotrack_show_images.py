# Small visualization tool. Keeps only one TIFF file in memory at a time, so it should start up fast and have a
# reasonable low memory footprint.
from config import ConfigFile
from imaging import tifffolder, image_visualizer, Experiment
from matplotlib import pyplot
from os import path

# PARAMETERS
config = ConfigFile("image_viewer")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
config.save_if_changed()
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format)
image_visualizer.show(experiment)

print("Done!")
pyplot.show()
