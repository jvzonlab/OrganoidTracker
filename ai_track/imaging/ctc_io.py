"""IO functions for the Cell Tracking Challenge data format."""
import os
from typing import Optional, Dict, List
from numpy import ndarray

import mahotas
from tifffile import tifffile

from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.core.position import Position
from ai_track.core.position_collection import PositionCollection


class _Link:
    daughter_id: int
    parent_id: int

    def __init__(self, daughter_id: int, parent_id: int):
        self.daughter_id = daughter_id
        self.parent_id = parent_id


def _read_lineage_file(file_name: str) -> Dict[int, List[_Link]]:
    links_to_mother_by_time_point = dict()

    with open(file_name) as file:
        for line in file:
            label, start, end, parent_label = line.split(" ")
            if int(parent_label) == 0:
                continue  # Cell 0 doesn't exist, so this track appeared out of nowhere

            link = _Link(int(label), int(parent_label))
            connections = links_to_mother_by_time_point.get(int(start))
            if connections is None:
                connections = list()
                links_to_mother_by_time_point[int(start)] = connections
            connections.append(link)

    return links_to_mother_by_time_point


def _box_to_position(position_box: ndarray, time_point_number: int) -> Optional[Position]:
    if position_box.max() == 0:
        # Only zeroes
        return None

    x = (position_box[4] + position_box[5]) / 2
    y = (position_box[2] + position_box[3]) / 2
    z = (position_box[0] + position_box[1]) / 2

    return Position(x, y, z, time_point_number=time_point_number)


def load_data_file(file_name: str, min_time_point: int = 0, max_time_point: int = 5000, *,
                   experiment: Optional[Experiment] = None) -> Experiment:
    if not file_name.lower().endswith(".txt"):
        raise ValueError("Can only load from .txt file")
    if experiment is None:
        experiment = Experiment()

    all_positions = PositionCollection()
    all_links = Links()
    links_to_mother_by_time_point = _read_lineage_file(file_name)

    file_prefix = file_name[:-4]
    time_point_number = 0
    positions_of_previous_time_point = list()

    while os.path.exists(f"{file_prefix}{time_point_number:03}.tif"):
        if time_point_number < min_time_point:
            continue
        if time_point_number > max_time_point:
            break

        print(f"Working on time point {time_point_number}...")
        image = tifffile.imread(f"{file_prefix}{time_point_number:03}.tif")
        position_boxes : ndarray = mahotas.labeled.bbox(image)
        positions_of_time_point = [None]  # ID 0 is never used, that's the background

        for i in range(1, position_boxes.shape[0]):
            position = _box_to_position(position_boxes[i], time_point_number)
            positions_of_time_point.append(position)

            if position is None:
                continue

            # Add position
            all_positions.add(position)
            all_links.set_position_data(position, "ctc_id", i)

            # Try to add link
            if i < len(positions_of_previous_time_point):
                previous_position = positions_of_previous_time_point[i]
                if previous_position is not None:
                    all_links.add_link(position, positions_of_previous_time_point[i])

        # Add mother-daughter links
        links_to_mother = links_to_mother_by_time_point.get(time_point_number)
        if links_to_mother is not None:
            for link in links_to_mother:
                daughter = positions_of_time_point[link.daughter_id]
                mother = positions_of_previous_time_point[link.parent_id]
                all_links.add_link(mother, daughter)

        time_point_number += 1
        positions_of_previous_time_point = positions_of_time_point

    experiment.positions = all_positions
    experiment.links = all_links

    return experiment
