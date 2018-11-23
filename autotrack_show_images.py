#!/usr/bin/env python3

# Small visualization tool. Keeps only one TIFF file in memory at a time, so it should start up fast and have a
# reasonable low memory footprint.

from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.gui import launcher
from autotrack.imaging import tifffolder
from autotrack.visualizer import standard_image_visualizer

# PARAMETERS
config = ConfigFile("image_viewer")
_images_folder = config.get_or_prompt("images_folder", "In which folder are the images stored? (Use ./ for the current "
                                                       "folder and use ../ for one level up.)", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format)
standard_image_visualizer.show(experiment)

print("Done!")
launcher.mainloop()
