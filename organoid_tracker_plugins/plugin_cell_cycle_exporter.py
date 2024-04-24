
import json
import csv
import os
from typing import Dict, Any, List, Optional

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
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.linking_analysis import lineage_markers
from organoid_tracker.linking_analysis.cell_fate_finder import CellFateType


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export cell cycle info":
            lambda: _export_cell_cycle_as_csv(window)
    }


def _export_cell_cycle_as_csv(window: Window):
    experiment = window.get_experiment()
    if not experiment.positions.has_positions():
        raise UserError("No positions are found", "No annotated positions are found. Cannot export anything.")

    folder = dialog.prompt_save_file("Select a directory", [("Folder", "*")])
    if folder is None:
        return
    os.mkdir(folder)
    positions = experiment.positions

    file_prefix = experiment.name.get_save_name() + ".csv"
    file_name = os.path.join(folder, file_prefix)

    for link in experiment.links.find_all_links():
        division_penalty1 = experiment.position_data.get_position_data(link[0], 'division_penalty')
        division_penalty2 = experiment.position_data.get_position_data(link[1], 'division_penalty')

        next_position = experiment.links.find_single_future(link[1])
        if (next_position is not None) and (division_penalty1 is not None):
            division_penalty3 = experiment.position_data.get_position_data(next_position, 'division_penalty')

            if (division_penalty1 + division_penalty2 + division_penalty3) / 3 < -2.0:
                track = experiment.links.get_track(link[0])
                if ((link[0].time_point_number() - track.first_time_point_number() > 6)
                        and (track.last_time_point_number() - link[0].time_point_number() > 6)):
                    experiment.links.remove_link(link[1], next_position)

    with open(file_name, "w", newline='') as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(("id", "parent_id", "sister_id", "start", "end",
                             "division_penalty_start", "division_penalty_end",
                             "start_division", "end_division"))

        for track in experiment.links.find_all_tracks():

            id = experiment.links.get_track_id(track)

            first_pos = track.find_first_position()
            last_pos = track.find_last_position()

            start = first_pos.time_point_number()
            end = last_pos.time_point_number()

            division_penalty_start = experiment.position_data.get_position_data(first_pos, "division_penalty")
            division_penalty_end = experiment.position_data.get_position_data(last_pos, "division_penalty")

            if division_penalty_end is None:
                division_penalty_end = 1000
            if division_penalty_start is None:
                division_penalty_start = 1000

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

            daughter_tracks = track.get_next_tracks()

            if (len(daughter_tracks) > 1) or (division_penalty_end < 0):
                end_division = end
            else:
                end_division = None

            csv_writer.writerow((id, parent_id, sister_id, start, end,
                                 division_penalty_start, division_penalty_end,
                                 start_division, end_division))

    dialog.popup_message("Positions", "Exported all positions as CSV files.")
