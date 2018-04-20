import os
import pickle
import re
import sys
import json

import math

from imaging import Experiment, io


def extract_positions(tracks_dir: str, output_file: str, min_time_point: int = 0, max_time_point: int = 5000) -> None:
    """Extracts all positions from the track files in tracks_dir, saves them to output_dir"""

    _fix_python_path_for_pickle()
    all_positions = Experiment()
    track_files = os.listdir(tracks_dir)
    print("Found " + str(len(track_files)) + " files to analyse")

    file_index = 0
    for track_file in track_files:
        if (file_index % 10) == 0:
            print("Working on track file " + str(file_index) + "/" + str(len(track_files)))

        # Check if file is actually a track file
        match = re.search('track_(\d\d\d\d\d)', track_file)
        if match:
            _extract_positions_from_track(os.path.join(tracks_dir, track_file), all_positions,
                                          min_time_point=min_time_point, max_time_point=max_time_point)

        file_index += 1

    io.save_positions_to_json(all_positions, output_file)


def _fix_python_path_for_pickle():
    # Inserts the current directory once to the Python module path
    # This makes Pickle able to find the necessary modules
    path = os.path.dirname(os.path.abspath(__file__))
    try :
        sys.path.index(path)
    except ValueError:
        print("Adding to Python path: " + path)
        sys.path.insert(0, path)


def _extract_positions_from_track(track_file: str, all_positions: Experiment, min_time_point: int = 0,
                                  max_time_point: int = 5000):
    """Extracts the tracks from a single file"""
    with open(track_file, "rb") as file_handle:
        track = pickle.load(file_handle, encoding = 'latin1')
        for time_point in track.t:
            if time_point < min_time_point or time_point > max_time_point:
                continue
            pos = track.get_pos(time_point)
            if math.isnan(pos[0] + pos[1] + pos[2]):
                print("Warning: encountered invalid position: " + str(pos))
                continue
            all_positions.add_particle_raw(pos[0], pos[1], pos[2], time_point)

