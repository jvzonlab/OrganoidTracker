# Simple nearest-neighbor linking
from imaging import Experiment, positions, tifffolder, visualizer
from nearest_neighbor_linking import tree_creator
import networkx
from matplotlib import pyplot


# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_positions_file = "../Results/" + _name + "/Manual positions.json"
_images_folder = "../Images/" + _name + "/"
_images_format= "nd799xy08t%03dc1.tif"
_max_frame = 5000
# END OF PARAMETERS


experiment = Experiment()
print("Loading cell positions...")
positions.load_positions_from_json(experiment, _positions_file)
print("Disovering images")
tifffolder.load_images_from_folder(experiment, _images_folder, _images_format, max_frame=_max_frame)
print("Staring link process")
tree_creator.link_particles(experiment, max_frame=_max_frame)
#networkx.draw(graph, with_labels=True, font_weight='bold', node_size=3)
print("Visualizing")
vis = visualizer.visualize(experiment)
print("Done")
pyplot.show()
print(vis)
