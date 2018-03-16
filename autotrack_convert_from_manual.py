#!/usr/bin/env python3

# This script is used to convert data from Guizela's scripts (e.g. track_manually.py) to data in our standard JSON
# format. Just launch it

from manual_tracking import positions_extractor, links_extractor
from imaging import io
from configparser import ConfigParser

# CONFIGURATION
config_file = "autotrack.ini"
config_changed = False
print("Hi! Configuration file is stored at " + config_file)
config = ConfigParser()
config.read(config_file)
if "DEFAULT" not in config:
    config["DEFAULT"] = {}
if "convert_from_manual" not in config:
    config["convert_from_manual"] = {}
    config_changed = True
our_section = config["convert_from_manual"]
if "image_folder" not in our_section:
    config["DEFAULT"]["image_folder"] = "."
    config_changed = True
if "min_time_point" not in our_section:
    config["DEFAULT"]["min_time_point"] = str(1)
    config_changed = True
if "max_time_point" not in our_section:
    config["DEFAULT"]["max_time_point"] = str(9999)
    config_changed = True
if "positions_file" not in our_section:
    our_section["positions_file"] = "Automatic analysis/Positions/Manual.json"
    config_changed = True
if "links_file" not in our_section:
    our_section["links_file"] = "Automatic analysis/Links/Manual.json"
    config_changed = True
if "input_tracks_folder" not in our_section:
    our_section["input_tracks_folder"] = input("Please enter the name of the folder where the track_xxxxx.p files are stored:")
    config_changed = True
if config_changed:
    with open(config_file, 'w') as config_writing:
        config.write(config_writing)
    if input("Configuration file was changed. Type start to run the script now.") != "start":
        print("Typed something else, bye!")
        exit(0)
    print("Ok, let's start!")
# END OF CONFIGURATION


positions_extractor.extract_positions(our_section["input_tracks_folder"], our_section["positions_file"],
                                      min_time_point=int(our_section["min_time_point"]),
                                      max_time_point=int(our_section["max_time_point"]))
io.save_links_to_json(
                      links_extractor.extract_from_tracks(our_section["input_tracks_folder"],
                                                        min_time_point=int(our_section["min_time_point"]),
                                                        max_time_point=int(our_section["max_time_point"])),
                      our_section["links_file"])
print("Done!")
