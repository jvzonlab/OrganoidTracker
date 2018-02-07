from nearest_neighbor_linking import positions
from pprint import pprint

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_input_file = "../Results/" + _name + "/Manual positions.json"
# END OF PARAMETERS

particles = positions.load_positions_from_json(_input_file)
