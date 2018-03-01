from imaging import Experiment, io, tifffolder
from linking_analysis import link_fixer, linker_for_experiment

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Positions/Manual.json"
_output_file = "../Results/" + _name + "/Smart nearest neighbor links.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_time_point = 0
_max_time_point = 115  # Organoid moved position here
_mitotic_radius = 3
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions...")
io.load_positions_from_json(experiment, _positions_file)
print("Discovering images...")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format,
                                   min_time_point=_min_time_point, max_time_point=_max_time_point)
print("Performing nearest-neighbor linking...")
possible_links = linker_for_experiment.link_particles(experiment, min_time_point=_min_time_point, max_time_point=_max_time_point, tolerance=2)
print("Deciding on what links to use...")
link_result = link_fixer.prune_links(experiment, possible_links, _mitotic_radius)
print("Writing results to file...")
io.save_links_to_json(link_result, _output_file)
print("Done")
