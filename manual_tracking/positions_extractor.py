import os
import pickle
import re
import sys
import json


class Positions:
    """Keeps track of all cell positions in a photo series"""
    _pos_per_frame = {}

    def add_pos(self, x, y, z, t):
        pos_list = []
        try:
            pos_list = self._pos_per_frame[str(t)]
        except KeyError:
            # Insert new list instead
            self._pos_per_frame[str(t)] = pos_list
        pos_list.append((x,y,z))

    def write(self, file):
        """Writes all positions as JSON"""
        print("Writing positions...")
        with open(file, 'w') as handle:
            json.dump(self._pos_per_frame, handle)
        print("Written all positions to " + file)


def extract_positions(tracks_dir : str, output_file : str):
    """Extracts all positions from the track files in tracks_dir, saves them to output_dir"""

    _fix_python_path_for_pickle()
    all_positions = Positions()
    track_files = os.listdir(tracks_dir)
    print("Found " + str(len(track_files)) + " files to analyse")

    file_index = 0
    for track_file in track_files:
        if (file_index % 10) == 0:
            print("Working on track file " + str(file_index) + "/" + str(len(track_files)))

        # Check if file is actually a track file
        match = re.search('track_(\d\d\d\d\d)', track_file)
        if match:
            _extract_positions_from_track(os.path.join(tracks_dir, track_file), all_positions)

        file_index += 1

    all_positions.write(output_file)


def _fix_python_path_for_pickle():
    # Inserts the current directory once to the Python module path
    # This makes Pickle able to find the necessary modules
    path = os.path.dirname(os.path.abspath(__file__))
    try :
        sys.path.index(path)
    except ValueError:
        print("Adding to Python path: " + path)
        sys.path.insert(0, path)


def _extract_positions_from_track(track_file : str, all_positions : Positions):
    """Extracts the tracks from a single file"""
    with open(track_file, "rb") as file_handle:
        track = pickle.load(file_handle, encoding = 'latin1')
        for frame in track.t:
            pos = track.get_pos(frame)
            all_positions.add_pos(pos[0], pos[1], pos[2], frame)

