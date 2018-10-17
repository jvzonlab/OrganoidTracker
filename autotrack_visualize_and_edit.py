#!/usr/bin/env python3

# Compares two linking results. The baseline links are assumed to be 100% correct, any deviations from that are
# counted as errors. Solid lines in the figures represent the correct linking result, dotted lines any deviations from
# that.
from os import path

from autotrack import gui
from autotrack.config import ConfigFile
from autotrack.core.links import LinkType
from autotrack.imaging import tifffolder, io
from autotrack.visualizer import image_visualizer
from autotrack.core.experiment import Experiment

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("visualize_and_edit")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_links_file = config.get_or_default("links_file", "Automatic analysis/Links/Smart nearest neighbor.json")
config.save_and_exit_if_changed()
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
if path.exists(_positions_file):
    io.load_positions_and_shapes_from_json(experiment, _positions_file,
                                           min_time_point=_min_time_point, max_time_point=_max_time_point)
if path.exists(_links_file):
    io.load_linking_result(experiment, _links_file, LinkType.BASELINE)

tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
vis = image_visualizer.show(experiment)

print("Done!")
gui.mainloop()
