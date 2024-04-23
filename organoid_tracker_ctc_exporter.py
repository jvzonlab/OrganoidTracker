
import json
import csv
import os
from typing import Dict, Any, List, Optional

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.imaging import io
from organoid_tracker.imaging.ctc_io import save_data_files

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("ctc_export")


_images_folder = config.get_or_prompt("images_container", "If you have a folder of image files, please paste the folder"
                                                          " path here. Else, if you have a LIF file, please paste the path to that file"
                                                          " here.", store_in_defaults=True)
_images_format = config.get_or_prompt("images_pattern", "What are the image file names? (Use {time:03} for three digits"
                                                        " representing the time point, use {channel} for the channel)",
                                      store_in_defaults=True)
_positions_file = config.get_or_prompt("positions_file",
                                        "What are the detected positions for those images?")

_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))

_output_folder = config.get_or_default("output_folder", "_RES", comment="Output file for the cell cycles.")
_mask_size_um = float(config.get_or_default("mask_size_um", str(7)))

config.save_and_exit_if_changed()

print("Load experiment")
experiment = io.load_data_file(_positions_file, _min_time_point, _max_time_point)
general_image_loader.load_images(experiment, _images_folder, _images_format,
                                 min_time_point=_min_time_point, max_time_point=_max_time_point)

print("start saving")
save_data_files(experiment, _output_folder, mask_size_um=_mask_size_um)

print("Exported all positions")
