# Compares two linking results. The baseline links are assumed to be 100% correct, any deviations from that are
# counted as errors. Solid lines in the figures represent the correct linking result, dotted lines any deviations from
# that.
from config import ConfigFile
from imaging import io, tifffolder, image_visualizer, Experiment
from linking_analysis import comparison
from matplotlib import pyplot

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
config.save_if_changed()

_automatic_links = io.load_links_from_json(_automatic_links_file)
_baseline_links = io.load_links_from_json(_baseline_links_file)
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
