#!/usr/bin/env python3

# This script is used to convert data from Guizela's scripts (e.g. track_manually.py) to data in our standard JSON
# format. Just launch it
from config import ConfigFile
from core import io
from manual_tracking import positions_extractor, links_extractor

# CONFIGURATION
print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("convert_from_manual")
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))
_positions_file = config.get_or_default("positions_file", "Automatic analysis/Positions/Manual.json")
_links_file = config.get_or_default("links_file", "Automatic analysis/Links/Manual.json")
_tracks_folder = config.get_or_prompt("input_tracks_folder", "Please enter the name of the folder where the track_xxxxx.p files are stored:")
if config.save_if_changed():
    print("Note: the configuration file was changed. Starting script...")
# END OF CONFIGURATION


positions_extractor.extract_positions(_tracks_folder, _positions_file, min_time_point=_min_time_point,
                                      max_time_point=_max_time_point)
io.save_links_to_json(links_extractor.extract_from_tracks(_tracks_folder, min_time_point=_min_time_point,
                                                          max_time_point=_max_time_point), _links_file)
print("Done!")
