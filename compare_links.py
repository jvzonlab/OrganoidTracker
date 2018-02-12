# Compares two linking results. The baseline links are assumed to be 100% correct, any deviations from that are
# counted as errors. Solid lines in the figures represent the correct linking result, dotted lines any deviations from
# that.

from imaging import io, tifffolder, Experiment
from imaging import image_visualizer
from matplotlib import pyplot
import networkx

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Manual positions.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"

_automatic_links = io.load_links_from_json("../Results/" + _name + "/Smart nearest neighbor links.json")
_baseline_links = io.load_links_from_json("../Results/" + _name + "/Manual links.json")
# END OF PARAMETERS

print("Starting...")

experiment = Experiment()
io.load_positions_from_json(experiment, _positions_file)
experiment.particle_links(_automatic_links)
experiment.particle_links_baseline(_baseline_links)

print("There are " + str(networkx.number_of_edges(_baseline_links)) + " connections in the baseline results.")
missed_links = networkx.difference(_baseline_links, _automatic_links);
made_up_links = networkx.difference(_automatic_links, _baseline_links);

print("There are " + str(networkx.number_of_edges(missed_links)) + " connections missed in the automatic results:")
for particle1, particle2 in missed_links.edges():
    print("\t" + str(particle1) + "---" + str(particle2))

print("There are " + str(networkx.number_of_edges(made_up_links)) + " connections made by the automatic linker, that"
      " did not exist in the manual results:")
for particle1, particle2 in made_up_links.edges():
    print("\t" + str(particle1) + "---" + str(particle2))

tifffolder.load_images_from_folder(experiment, _images_folder, _images_format)
vis = image_visualizer.show(experiment)

print("Done!")
pyplot.show()
