
import json
import csv
import os
from typing import Dict, Any, List, Optional

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import io
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.linking_analysis import lineage_markers
from organoid_tracker.linking_analysis.cell_fate_finder import CellFateType

print("Hi! Configuration file is stored at " + ConfigFile.FILE_NAME)
config = ConfigFile("cell_cycle_export")


_experiment_file = config.get_or_prompt('positions_file', 'Where are the filtered tracks stored?')
_min_time_point = int(config.get_or_default("min_time_point", str(1), store_in_defaults=True))
_max_time_point = int(config.get_or_default("max_time_point", str(9999), store_in_defaults=True))

_output_file = config.get_or_default("output_file", "cell_cycles.csv", comment="Output file for the cell cycles.")

config.save_and_exit_if_changed()

experiment = io.load_data_file(_experiment_file, _min_time_point, _max_time_point)

if not experiment.positions.has_positions():
    raise UserError("No positions are found", "No annotated positions are found. Cannot export anything.")

positions = experiment.positions

# check for missed divisions in the tracks
for link in experiment.links.find_all_links():
    division_penalty1 = experiment.positions.get_position_data(link[0], 'division_penalty')
    division_penalty2 = experiment.positions.get_position_data(link[1], 'division_penalty')

    next_position = experiment.links.find_single_future(link[1])
    if (next_position is not None) and (division_penalty1 is not None):
        division_penalty3 = experiment.positions.get_position_data(next_position, 'division_penalty')

        if (division_penalty1 + division_penalty2 + division_penalty3)/3 < -2.0:
            track = experiment.links.get_track(link[0])
            if ((link[0].time_point_number() - track.first_time_point_number() > 6)
                and (track.last_time_point_number() - link[0].time_point_number() > 6)):
                print(experiment.links.contains_link(link[1], next_position))
                experiment.links.remove_link(link[1], next_position)
                print(experiment.links.contains_link(link[1], next_position))
                print(link[0])

with open(_output_file, "w", newline='') as handle:
    csv_writer = csv.writer(handle)
    csv_writer.writerow(("id", "parent_id", "sister_id", "cousin1_id", "cousin2_id", "start", "end",
                             "division_penalty_start", "division_penalty_end",
                             "start_division", "end_division", "last_x", "last_y", "last_z"))

    for track in experiment.links.find_all_tracks():

        id = experiment.links.get_track_id(track)

        first_pos = track.find_first_position()
        last_pos = track.find_last_position()

        start = first_pos.time_point_number()
        end = last_pos.time_point_number()

        division_penalty_start = experiment.positions.get_position_data(first_pos, "division_penalty")
        division_penalty_end = experiment.positions.get_position_data(last_pos, "division_penalty")

        if division_penalty_start is None:
            division_penalty_start = 1000
        if division_penalty_end is None:
            division_penalty_end = 1000

        parent_track = list(track.get_previous_tracks())

        if len(parent_track) > 0:
            parent_id = experiment.links.get_track_id(parent_track[0])
            start_division = start

            daughter1, daughter2 = parent_track[0].get_next_tracks()
            sister_id = experiment.links.get_track_id(daughter1)
            if sister_id == id:
                sister_id = experiment.links.get_track_id(daughter2)

        elif division_penalty_start < 0:
            parent_id = None
            sister_id = None
            start_division = start

        else:
            parent_id = None
            sister_id = None
            start_division = None

        cousin1_id = None
        cousin2_id = None

        if len(parent_track) > 1:
            parent_aunt1, parent_aunt2 = parent_track[1].get_next_tracks()

            if experiment.links.get_track_id(parent_aunt1) == parent_id:
                aunt = parent_aunt2
            else:
                aunt = parent_aunt1

            cousins = aunt.get_next_tracks()

            if len(cousins)==1:
                cousin1_id = experiment.links.get_track_id(cousins[0])
                cousin2_id = None
            elif len(cousins)==2:
                cousin1_id = experiment.links.get_track_id(cousins[0])
                cousin2_id = experiment.links.get_track_id(cousins[1])

        daughter_tracks = track.get_next_tracks()

        if (len(daughter_tracks) > 1) or (division_penalty_end < 0):
            end_division = end
        else:
            end_division = None

        csv_writer.writerow((id, parent_id, sister_id, start, end,
                                 division_penalty_start, division_penalty_end,
                                 start_division, end_division, last_pos.x, last_pos.y, last_pos.z))

print("Exported all positions as CSV files.")
