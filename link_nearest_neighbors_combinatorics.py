from matplotlib import pyplot

from imaging import Experiment, io, tifffolder, image_visualizer
from linking import link_fixer_combinatorics, linker_for_experiment
from linking_analysis import mother_finder
import time

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Positions/Manual.json"
_output_file = "../Results/" + _name + "/Combinatorics nearest neighbor links.json"
_comparison_links_file = "../Results/" + _name + "/Manual links.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_time_point = 0
_max_time_point = 115
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
link_result = linker_for_experiment.link_particles(experiment, min_time_point=_min_time_point, max_time_point=_max_time_point, tolerance=2)
print("Deciding on what links to use...")

start_time = time.time()
link_result, families = link_fixer_combinatorics.prune_links(experiment, link_result, _detection_radius_small,
                                                   _detection_radius_large, _max_distance)
print("Time elapsed: {:.2f}s".format(time.time() - start_time))
print("Writing results to file...")
io.save_links_to_json(link_result, _output_file)

print("Basic comparison...")
baseline_links = io.load_links_from_json(_comparison_links_file)
families_baseline = mother_finder.find_families(baseline_links)

missed_families = families_baseline.difference(families)
print("Missed families: " + str(len(missed_families)))
for missed_family in missed_families:
    print("\t" + str(missed_family))

made_up_families = families.difference(families_baseline)
print("Made up families: " + str(len(made_up_families)))
for made_up_family in made_up_families:
    print("\t" + str(made_up_family))


print("Done!")
