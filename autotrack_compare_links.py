#!/usr/bin/env python3

# Compares two linking results. The baseline links are assumed to be 100% correct, any deviations from that are
# counted as errors. Solid lines in the figures represent the correct linking result, dotted lines any deviations from
# that.
from autotrack import gui
from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.linking_analysis import comparison
from autotrack.visualizer import image_visualizer

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("compare_links")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_automatic_links_file = config.get_or_default("automatic_links_file",
                                              "Automatic analysis/Links/Smart nearest neighbor.json")
_baseline_links_file = config.get_or_default("baseline_links_file", "Automatic analysis/Links/Manual.json")
config.save_and_exit_if_changed()

# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
io.load_positions_and_shapes_from_json(experiment, _positions_file)
io.load_links_and_scores_from_json(experiment, _automatic_links_file, links_are_scratch=True)
experiment.particle_links(io.load_links_from_json(_baseline_links_file,
                                                  min_time_point=_min_time_point, max_time_point=_max_time_point))

comparison.print_differences(experiment)

tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
vis = image_visualizer.show(experiment)

print("Done!")
gui.mainloop()
