from nearest_neighbor_linking import positions
from nearest_neighbor_linking import tree_creator
import networkx
from matplotlib import pyplot


# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_input_file = "../Results/" + _name + "/Manual positions.json"
# END OF PARAMETERS


particles = positions.load_positions_from_json(_input_file)
graph = tree_creator.link_particles(particles)
networkx.draw(graph, with_labels=False, font_weight='bold', node_size=3)
pyplot.show()
