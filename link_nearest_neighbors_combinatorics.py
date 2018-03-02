from matplotlib import pyplot

from imaging import Experiment, io, tifffolder, image_visualizer
from linking import link_fixer_combinatorics, linker_for_experiment
from linking_analysis import mother_finder

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Positions/Manual.json"
_output_file = "../Results/" + _name + "/Smart nearest neighbor links.json"
_comparison_links_file = "../Results/" + _name + "/Manual links.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_time_point = 0
_max_time_point = 5000
_detection_radius_large = 10  # Used to check for cells that are maybe a mother/daughter
_detection_radius_small = 2  # Used to check for cells that are surely a mother/daughter
_max_distance = 40  # Maximum distance between centers of a mother and a daughter in pixels
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
mothers, daughters = link_fixer_combinatorics.prune_links(experiment, possible_links, _detection_radius_small,
                                                   _detection_radius_large, _max_distance)
print("Writing results to file...")
baseline_links = io.load_links_from_json(_comparison_links_file)
mothers_baseline, daughters_baseline = mother_finder.find_mothers_and_daughters(baseline_links)

mothers_baseline.difference_update(mothers)
print("Missed mothers: " + str(mothers_baseline))
daughters_baseline.difference_update(daughters)
print("Missed daughters: " + str(daughters_baseline))

#io.save_links_to_json(link_result, _output_file)
print("Done!")
