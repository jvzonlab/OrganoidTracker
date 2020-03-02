"""IO functions for the Cell Tracking Challenge data format, following the specification at
https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf """
import os
from typing import Optional, Dict, List

import numpy
from numpy import ndarray

import mahotas
from tifffile import tifffile

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection


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


def save_data_files(experiment: Experiment, file_name: str):
    """Saves all cell tracks in the data format of the Cell Tracking Challenge. Requires the presence of links and
     images. Also requires an image size to be known, as well as the file name ending with .txt (case insensitive).
     Throws ValueError if any of these conditions are violated."""
    if not file_name.lower().endswith(".txt"):
        raise ValueError("Must save as a .txt file")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    image_size_zyx = experiment.images.image_loader().get_image_size_zyx()
    if image_size_zyx is None:
        raise ValueError("Couldn't find an image size.")
    if not experiment.links.has_links():
        raise ValueError("No links found")

    image_prefix = file_name[0:-4]
    _save_track_images(experiment, image_prefix)
    _save_overview_file(experiment, file_name)


def _save_track_images(experiment: Experiment, image_prefix: str):
    image_size_zyx = experiment.images.image_loader().get_image_size_zyx()
    links = experiment.links
    positions = experiment.positions
    offsets = experiment.images.offsets

    position_half_width_px = 2  # Positions will have a visible size of 2x + 1
    for time_point in positions.time_points():
        image_file_name = f"{image_prefix}{time_point.time_point_number():03}.tif"
        image_array = numpy.zeros(image_size_zyx, dtype=numpy.uint16)
        image_offset = offsets.of_time_point(time_point)

        for position in positions.of_time_point(time_point):
            moved_position = position - image_offset
            x, y, z = int(moved_position.x), int(moved_position.y), int(moved_position.z)

            # Only export positions in the images
            if x < position_half_width_px or x >= image_size_zyx[2] - position_half_width_px:
                continue
            if y < position_half_width_px or y >= image_size_zyx[1] - position_half_width_px:
                continue
            if z < 0 or z >= image_size_zyx[0]:
                continue
            track = links.get_track(position)
            if track is None:
                continue  # No links, so we cannot save the position

            track_id = links.get_track_id(track)
            image_array[z,
            y - position_half_width_px: y + position_half_width_px + 1,
            x - position_half_width_px: x + position_half_width_px + 1] = track_id
        tifffile.imsave(image_file_name, image_array, compress=9)


def _save_overview_file(experiment: Experiment, file_name: str):
    """Save overview of all linking tracks and their ids"""

    links = experiment.links
    with open(file_name, "w") as handle:
        for track_id, track in experiment.links.find_all_tracks_and_ids():
            parent_id = 0
            previous_tracks = track.get_previous_tracks()
            if len(previous_tracks) == 1:
                parent_id = links.get_track_id(previous_tracks.pop())
                if parent_id is None:
                    parent_id = 0

            handle.write(
                f"{track_id} {track.min_time_point_number()} {track.max_time_point_number()} {parent_id}\n")
