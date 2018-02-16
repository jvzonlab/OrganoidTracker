# This script is used to extract x,y,z positions of all cells for every time point from the manual tracking data
# Input: directory of track_xxxxx.p files
# Output: single JSON file

from manual_tracking import positions_extractor, links_extractor
from imaging import io

# PARAMETERS
_name = "multiphoton.organoids.17-07-28_weekend_H2B-mCherry.nd799xy08"
_input_dir = "../Results/" + _name + "/Manual tracks/"
_output_file_positions = "../Results/" + _name + "/Manual positions.json"
_output_file_tracks = "../Results/" + _name + "/Manual links.json"
_min_frame = 116  # Before this frame, the whole crypt has moved
_max_frame = 5000
# END OF PARAMETERS


positions_extractor.extract_positions(_input_dir, _output_file_positions, min_frame=_min_frame, max_frame=_max_frame)
io.save_links_to_json(
    links_extractor.extract_from_tracks(_input_dir, min_frame=_min_frame, max_frame=_max_frame),
    _output_file_tracks)
