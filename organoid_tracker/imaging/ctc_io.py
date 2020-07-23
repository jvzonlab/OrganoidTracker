"""IO functions for the Cell Tracking Challenge data format, following the specification at
https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf """
import math
import os
from typing import Optional, Dict, List

import numpy
from numpy import ndarray

import mahotas
from scipy.ndimage import distance_transform_edt
from tifffile import tifffile

from organoid_tracker.core import UserError, bounding_box
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.mask import Mask
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.util import bits


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
    all_position_data = PositionData()
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
            all_position_data.set_position_data(position, "ctc_id", i)

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
    experiment.position_data = all_position_data

    return experiment


def _create_spherical_mask(radius_um: float, resolution: ImageResolution) -> Mask:
    """Creates a mask that is spherical in micrometers. If the resolution is not the same in the x, y and z directions,
    this sphere will appear as a spheroid in the images."""
    radius_x_px = math.ceil(radius_um / resolution.pixel_size_x_um)
    radius_y_px = math.ceil(radius_um / resolution.pixel_size_y_um)
    radius_z_px = math.ceil(radius_um / resolution.pixel_size_z_um)
    mask = Mask(bounding_box.ONE.expanded(radius_x_px, radius_y_px, radius_z_px))

    # Evaluate the spheroid function to draw it
    mask.add_from_function(lambda x, y, z:
                           x ** 2 / radius_x_px ** 2 + y ** 2 / radius_y_px ** 2 + z ** 2 / radius_z_px ** 2 <= 1)

    return mask


def save_data_files(experiment: Experiment, folder: str):
    """Saves all cell tracks in the data format of the Cell Tracking Challenge. Requires the presence of links and
     images. Also requires an image size to be known, as well as the file name ending with .txt (case insensitive).
     Throws ValueError if any of these conditions are violated."""
    is_ground_truth = folder.endswith("_GT")
    is_scratch = folder.endswith("_RES")
    if not is_ground_truth and not is_scratch:
        raise UserError("Invalid folder name", "Folder name should end with \"_GT\" or \"_RES\", depending on whether"
                                               " the active dataset represent ground truth or tracked data.")
    image_size_zyx = experiment.images.image_loader().get_image_size_zyx()
    if image_size_zyx is None:
        raise ValueError("Couldn't find an image size.")
    if not experiment.links.has_links():
        raise ValueError("No links found")

    # Create mask for stamping the positions in the images
    if is_scratch:
        mask = _create_spherical_mask(5, experiment.images.resolution())
    else:
        mask = Mask(bounding_box.ONE.expanded(2, 2, 0))
        mask.get_mask_array().fill(1)

    # Create folder
    sub_folder = os.path.join(folder, "TRA") if is_ground_truth else folder
    os.makedirs(sub_folder, exist_ok=True)

    image_prefix = "man_track" if is_ground_truth else "mask"
    if is_scratch:
        resolution = experiment.images.resolution()
        _save_track_images_watershed(experiment, os.path.join(sub_folder, image_prefix), mask, resolution)
    else:
        _save_track_images(experiment, os.path.join(sub_folder, image_prefix), mask)

    file_name = os.path.join("man_track.txt") if is_ground_truth else "res_track.txt"
    _save_overview_file(experiment, os.path.join(sub_folder, file_name))


def _save_track_images(experiment: Experiment, image_prefix: str, mask: Mask):
    """Saves images colored with all tracks at the right location. Each track is marked using the given mask."""
    image_size_zyx = experiment.images.image_loader().get_image_size_zyx()
    links = experiment.links
    positions = experiment.positions
    offsets = experiment.images.offsets

    for time_point in positions.time_points():
        image_file_name = f"{image_prefix}{time_point.time_point_number():03}.tif"
        image_fill_array = numpy.zeros(image_size_zyx, dtype=numpy.uint16)
        image_offset = offsets.of_time_point(time_point)

        for position in positions.of_time_point(time_point):
            moved_position = position - image_offset

            track = links.get_track(position)
            if track is None:
                continue  # No links, so we cannot save the position

            track_id = links.get_track_id(track)
            mask.center_around(moved_position)
            mask.stamp_image(image_fill_array, track_id + 1)  # Track id is offset by 1 to avoid track id 0

        tifffile.imsave(image_file_name, image_fill_array, compress=9)


def _save_track_images_watershed(experiment: Experiment, image_prefix: str, mask: Mask, resolution: ImageResolution):
    """Saves images colored with all tracks at the right location. Each track is marked using the given mask. A
    watershed transformation is applied to handle overlapping masks."""
    image_size_zyx = experiment.images.image_loader().get_image_size_zyx()
    links = experiment.links
    positions = experiment.positions
    offsets = experiment.images.offsets

    for time_point in positions.time_points():
        image_file_name = f"{image_prefix}{time_point.time_point_number():03}.tif"
        image_mask_array = numpy.zeros(image_size_zyx, dtype=numpy.uint8)
        image_seed_array = numpy.zeros(image_size_zyx, dtype=numpy.uint16)
        image_offset = offsets.of_time_point(time_point)

        for position in positions.of_time_point(time_point):
            moved_position = position - image_offset

            track = links.get_track(position)
            if track is None:
                continue  # No links, so we cannot save the position

            track_id = links.get_track_id(track)
            mask.center_around(moved_position)
            mask.stamp_image(image_mask_array, 1)
            image_seed_array[int(moved_position.z), int(moved_position.y), int(moved_position.x)] = track_id + 1
            # ^ Track id is offset by 1 to avoid track id 0

        distance_map = distance_transform_edt(image_seed_array == 0, sampling=resolution.pixel_size_zyx_um)
        background_color = distance_map.max() + 1
        distance_map[image_mask_array == 0] = background_color

        regions = mahotas.cwatershed(distance_map, image_seed_array).astype(numpy.uint16)
        regions[image_mask_array == 0] = 0  # Remove background
        tifffile.imsave(image_file_name, regions, compress=9)


def _save_overview_file(experiment: Experiment, file_name: str):
    """Save overview of all linking tracks and their ids"""

    links = experiment.links
    with open(file_name, "w") as handle:
        for track_id, track in experiment.links.find_all_tracks_and_ids():
            parent_id = -1
            previous_tracks = track.get_previous_tracks()
            if len(previous_tracks) == 1:
                parent_id = links.get_track_id(previous_tracks.pop())
                if parent_id is None:
                    parent_id = -1

            # Write the line, adding 1 to all track ids to avoid track 0
            handle.write(
                f"{track_id + 1} {track.min_time_point_number()} {track.max_time_point_number()} {parent_id + 1}\n")
