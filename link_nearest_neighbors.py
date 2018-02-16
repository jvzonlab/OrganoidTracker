# Simple nearest-neighbor linking: every cell is linked to the nearest cell in the previous image. Physical and
# biological constraints are not taken into account.
from imaging import Experiment, io, tifffolder, image_visualizer
from nearest_neighbor_linking import tree_creator
from matplotlib import pyplot


# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Manual positions.json"
_output_file = "../Results/" + _name + "/Nearest neighbor links.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_min_frame = 116  # Organoid moved position here
_max_frame = 5000
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions...")
io.load_positions_from_json(experiment, _positions_file)
print("Disovering images")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format, min_frame=_min_frame,
                                   max_frame=_max_frame)
print("Staring link process")
results = tree_creator.link_particles(experiment, min_frame=_min_frame, max_frame=_max_frame)
print("Writing results to file")
io.save_links_to_json(results, _output_file)
print("Visualizing")
image_visualizer.show(experiment)
print("Done")
pyplot.show()

