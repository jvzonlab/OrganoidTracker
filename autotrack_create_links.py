from config import ConfigFile
from imaging import Experiment, io, tifffolder
from linking import link_fixer_casebycase, linker_for_experiment

# PARAMETERS
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("create_links")
_images_folder = config.get_or_default("images_folder", "./", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use %03d for a three digits "
                                                        "representing the time)", store_in_defaults=True)
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_output_file = config.get_or_default("links_output_file", "Automatic analysis/Links/Smart nearest neighbor.json")
_mitotic_radius = int(config.get_or_default("mitotic_radius", str(3)))
_shape_detection_radius = int(config.get_or_default("shape_detection_radius", str(16)))
_flow_detection_radius = int(config.get_or_default("flow_detection_radius", str(50)))
config.save_if_changed()
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions...")
io.load_positions_from_json(experiment, _positions_file)
print("Discovering images...")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Performing nearest-neighbor linking...")
possible_links = linker_for_experiment.nearest_neighbor(experiment, tolerance=2,
                                                       min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Improving links by analyzing local average movement...")
possible_links = linker_for_experiment.nearest_neighbor_using_flow(experiment, possible_links,
                                                                   flow_detection_radius=_flow_detection_radius,
                                                                   min_time_point=_min_time_point,
                                                                   max_time_point=_max_time_point)
print("Deciding on what links to use...")
link_result = link_fixer_casebycase.prune_links(experiment, possible_links, _mitotic_radius, _shape_detection_radius)
print("Writing results to file...")
io.save_links_to_json(link_result, _output_file)
print("Done")
