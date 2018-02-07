# This script is used to extract x,y,z positions of all cells for every time point from the manual tracking data
# Input: directory of track_xxxxx.p files
# Output: single JSON file

from manual_tracking import positions_extractor

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_input_dir = "../Results/" + _name + "/Manual tracks/"
_output_file = "../Results/" + _name + "/Manual positions.json"
# END OF PARAMETERS


positions_extractor.extract_positions(_input_dir, _output_file)
