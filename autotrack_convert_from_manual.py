#!/usr/bin/env python3

# This script is used to convert data from Guizela's scripts (e.g. track_manually.py) to data in our standard JSON
# format. Just launch it


from autotrack.config import ConfigFile
from autotrack.core.experiment import Experiment
from autotrack.imaging import tifffolder, io
from autotrack.manual_tracking import links_extractor
from autotrack.particle_detection import ellipse_detector_for_experiment

# CONFIGURATION
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("convert_from_manual")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_links_file = config.get_or_default("links_file", "Automatic analysis/Links/Manual.json")
_tracks_folder = config.get_or_prompt("input_tracks_folder", "Please enter the name of the folder where the track_xxxxx.p files are stored:")
_shape_detection_radius = int(config.get_or_default("shape_detection_radius", str(16)))
config.save_and_exit_if_changed()
# END OF CONFIGURATION


graph = links_extractor.extract_from_tracks(_tracks_folder, min_time_point=_min_time_point,
                                            max_time_point=_max_time_point)
experiment = Experiment()
for particle in graph.nodes():
    experiment.add_particle(particle)
experiment.particle_links(graph)

print("Performing rudimentary 2d shape detection...")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
ellipse_detector_for_experiment.detect_for_all(experiment, _shape_detection_radius)

io.save_positions_and_shapes_to_json(experiment, _positions_file)
io.save_links_to_json(graph, _links_file)

print("Done!")