"""Extracting the tracks as measured by Guizela to a Graph object"""

import math
import os
import pickle
import re
import sys
from typing import List, Optional, Dict

import numpy

from organoid_tracker.core import TimePoint
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.spline import SplineCollection, Spline
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.guizela_tracker_compatibility import cell_type_converter
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.linking_analysis.linking_markers import EndMarker
from organoid_tracker.guizela_tracker_compatibility.track_lib import Track


def _load_links(experiment: Experiment, tracks_dir: str, min_time_point: int = 0, max_time_point: int = 5000):
    """Extracts all positions and links from the track files in tracks_dir, returns them as a Graph."""

    _fix_python_path_for_pickle()
    links = experiment.links
    position_data = experiment.position_data

    # Read tracks and divisions for links
    tracks = _read_track_files(tracks_dir, experiment, min_time_point=min_time_point, max_time_point=max_time_point)
    _read_lineage_file(tracks_dir, links, tracks, min_time_point=min_time_point, max_time_point=max_time_point)

    # Also add as positions
    positions = experiment.positions
    for position in experiment.links.find_all_positions():
        positions.add(position)

    _read_deaths_file(tracks_dir, position_data, tracks, min_time_point=min_time_point, max_time_point=max_time_point)
    for cell_type, file_name in cell_type_converter.CELL_TYPE_TO_FILE.items():
        _read_cell_type_file(tracks_dir, position_data, tracks, cell_type, file_name=file_name)


def _load_crypt_axis(tracks_dir: str, positions: PositionCollection, paths: SplineCollection, min_time_point: int,
                     max_time_point: int):
    """Loads the axis of the crypt and saves it as a Path to the experiment. The offset of path will be set such that
    the bottom-most position has a crypt axis position of 0."""
    _fix_python_path_for_pickle()
    file = os.path.join(tracks_dir, "crypt_axes.p")
    if not os.path.exists(file):
        return  # No crypt axis stored

    print("Reading crypt axes file")
    paths.set_marker_name(1, "CRYPT", True)  # We will import 1 axis, and that one will be the crypt axis
    with open(file, 'rb') as file_handle:
        axes = pickle.load(file_handle, encoding="latin1")
        for axis in axes:
            if axis.t < min_time_point or axis.t > max_time_point:
                continue

            path = Spline()
            axis.x.reverse()  # Guizela's paths are defined in exactly the opposite way of what we want
            for position in axis.x:  # axis.x == [[x,y,z],[x,y,z],[x,y,z],...]
                path.add_point(position[0], position[1], position[2])

            time_point = TimePoint(axis.t)
            path.update_offset_for_positions(positions.of_time_point(time_point))
            paths.add_spline(time_point, path, None)


def add_data_to_experiment(experiment: Experiment, tracks_dir: str, min_time_point: int = 0, max_time_point: int = 500):
    """Adds all positions and links from the given folder to the experiment."""
    _load_links(experiment, tracks_dir, min_time_point, max_time_point)
    _load_crypt_axis(tracks_dir, experiment.positions, experiment.splines, min_time_point, max_time_point)


def _read_track_files(tracks_dir: str, experiment: Experiment, min_time_point: int = 0, max_time_point: int = 5000
                      ) -> Dict[int, Track]:
    """Adds all tracks to the graph, and returns the original tracks"""
    track_files = os.listdir(tracks_dir)
    print("Found " + str(len(track_files)) + " files to analyse")

    tracks = dict()
    track_file_pattern = re.compile(r"^track_([0-9]{5})\.p$")

    track_counter = 1
    for track_file in track_files:
        match = track_file_pattern.match(track_file)
        if match is None:
            continue
        track_index = int(match.group(1))
        track_file = os.path.join(tracks_dir, track_file)
        if not os.path.exists(track_file):
            break

        if track_counter % 10 == 0:
            print("Reading track " + str(track_counter))

        # Note that the first track will get id 0, the second id 1, etc. This is required for the lineages file
        tracks[track_index] = _extract_links_from_track(track_file, experiment, min_time_point=min_time_point, max_time_point=max_time_point)

        track_counter += 1

    return tracks


def _read_lineage_file(tracks_dir: str, links: Links, tracks: Dict[int, Track], min_time_point: int = 0,
                       max_time_point: int = 5000) -> None:
    """Connects the lineages in the graph based on information from the lineages.p file"""
    print("Reading lineages file")
    lineage_file = os.path.join(tracks_dir, "lineages.p")
    if not os.path.exists(lineage_file):
        return
    with open(lineage_file, "rb") as file_handle:
        lineages = pickle.load(file_handle, encoding='latin1')
        for lineage in lineages:
            if lineage[0] not in tracks or lineage[1] not in tracks or lineage[2] not in tracks:
                print("Skipping division", lineage, ", not all tracks are found (there are", len(tracks), "tracks)")
                continue
            mother_track = tracks[lineage[0]]
            child_track_1 = tracks[lineage[1]]
            child_track_2 = tracks[lineage[2]]

            first_time_point_after_division = numpy.amin(child_track_1.t)
            if first_time_point_after_division - 1 < min_time_point or first_time_point_after_division > max_time_point:
                continue

            mother_last_snapshot = _get_cell_in_time_point(mother_track, first_time_point_after_division - 1)
            child_1_first_snapshot = _get_cell_in_time_point(child_track_1, first_time_point_after_division)
            child_2_first_snapshot = _get_cell_in_time_point(child_track_2, first_time_point_after_division)

            if mother_last_snapshot is not None\
                    and child_1_first_snapshot is not None\
                    and child_2_first_snapshot is not None:
                links.add_link(mother_last_snapshot, child_1_first_snapshot)
                links.add_link(mother_last_snapshot, child_2_first_snapshot)


def _read_deaths_file(tracks_dir: str, position_data: PositionData, tracks_by_id: Dict[int, Track], min_time_point: int,
                      max_time_point: int):
    """Adds all marked cell deaths to the linking network."""
    _fix_python_path_for_pickle()
    file = os.path.join(tracks_dir, "dead_cells.p")
    if not os.path.exists(file):
        return  # No crypt axis stored

    print("Reading cell deaths file")
    with open(file, 'rb') as file_handle:
        dead_track_numbers = pickle.load(file_handle, encoding="latin1")
        for dead_track_number in dead_track_numbers:
            if dead_track_number not in tracks_by_id:
                print(f"Track with id {dead_track_number} was marked as dead, but no such track exists.")
                continue
            track = tracks_by_id[dead_track_number]
            last_position_time = track.t[-1]
            if last_position_time < min_time_point or last_position_time > max_time_point:
                continue
            last_position_position = track.x[-1]
            last_position = Position(*last_position_position, time_point_number=last_position_time)
            linking_markers.set_track_end_marker(position_data, last_position, EndMarker.DEAD)


def _read_cell_type_file(tracks_dir: str, position_data: PositionData, tracks_by_id: Dict[int, Track], cell_type: str, *,
                         file_name: str):
    """Adds all marked cell deaths to the linking network. If the file name is not specified, it is assumed to be
    cell_type.p ."""
    _fix_python_path_for_pickle()

    file = os.path.join(tracks_dir, file_name)
    if not os.path.exists(file):
        return  # No crypt axis stored

    print(f"Reading {file_name} for cells of type \"{cell_type.lower()}\"")
    with open(file, 'rb') as file_handle:
        typed_cell_numbers = pickle.load(file_handle, encoding="latin1")
        for typed_cell_number in typed_cell_numbers:
            if typed_cell_number not in tracks_by_id:
                print(f"Track of id {typed_cell_number} as marked as having type \"{cell_type.lower()}\", but no such"
                      f" track exists")
                continue
            track = tracks_by_id[typed_cell_number]
            for i in range(len(track.x)):
                position = Position(*track.x[i], time_point_number=track.t[i])
                linking_markers.set_position_type(position_data, position, cell_type.upper())


def _get_cell_in_time_point(track: Track, time_point_number: int) -> Optional[Position]:
    position_array = track.get_pos(time_point_number)
    if len(position_array) == 0:
        return None
    return Position(position_array[0], position_array[1], position_array[2], time_point_number=time_point_number)


def _extract_links_from_track(track_file: str, experiment: Experiment, min_time_point: int = 0,
                              max_time_point: int = 5000) -> Track:
    links = experiment.links
    positions = experiment.positions
    with open(track_file, "rb") as file_handle:
        track = pickle.load(file_handle, encoding='latin1')
        current_position = None

        for time_point in track.t:
            if time_point < min_time_point or time_point > max_time_point:
                continue

            previous_position = current_position
            current_position = _get_cell_in_time_point(track, time_point)
            if math.isnan(current_position.x + current_position.y + current_position.z):
                print("Warning: found invalid " + str(current_position))
                continue

            positions.add(current_position)
            if previous_position is not None:
                while previous_position.time_point_number() < current_position.time_point_number() - 1:
                    temp_position = previous_position.with_time_point_number(previous_position.time_point_number() + 1)
                    links.add_link(previous_position, temp_position)
                    previous_position = temp_position

                links.add_link(previous_position, current_position)

        return track


def _fix_python_path_for_pickle():
    # Inserts the current directory once to the Python module path
    # This makes Pickle able to find the necessary modules
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.index(path)
    except ValueError:
        print("Adding to Python path: " + path)
        sys.path.insert(0, path)
