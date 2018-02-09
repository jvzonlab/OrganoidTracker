# Simple nearest-neighbor linking
from imaging import Experiment, io, tifffolder, image_visualizer
from nearest_neighbor_linking import tree_creator
from matplotlib import pyplot


# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Manual positions.json"
_output_file = "../Results/" + _name + "/Nearest neighbor links.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_max_frame = 5000
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions...")
io.load_positions_from_json(experiment, _positions_file)
print("Disovering images")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format, max_frame=_max_frame)
print("Staring link process")
tree_creator.link_particles(experiment, max_frame=_max_frame)
print("Writing results to file")
io.save_links_to_json(experiment.particle_links(), _output_file)
print("Visualizing")
image_visualizer.show(experiment)
print("Done")
pyplot.show()

