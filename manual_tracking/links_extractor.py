"""Extracting the tracks as measured by Guizela to a Graph object"""

from imaging import Particle
from manual_tracking.track_lib import Track
from networkx import Graph
from typing import List
import os
import pickle
import sys
import numpy


def extract_from_tracks(tracks_dir: str, min_time_point: int = 0, max_time_point: int = 5000) -> Graph:
    """Extracts all positions from the track files in tracks_dir, saves them to output_dir"""

    _fix_python_path_for_pickle()
    graph = Graph()

    tracks = _read_track_files(tracks_dir, graph, min_time_point=min_time_point, max_time_point=max_time_point)
    _read_lineage_file(tracks_dir, graph, tracks, min_time_point=min_time_point, max_time_point=max_time_point)

    return graph


def _read_track_files(tracks_dir: str, graph: Graph, min_time_point: int = 0, max_time_point: int = 5000) -> List[Track]:
    """Adds all tracks to the graph, and returns the original tracks"""
    track_files = os.listdir(tracks_dir)
    print("Found " + str(len(track_files)) + " files to analyse")

    track_index = 0
    tracks = []
    while True:
        track_file = os.path.join(tracks_dir, "track_%05d.p" % track_index)
        if not os.path.exists(track_file):
            break

        if track_index % 10 == 0:
            print("Reading track " + str(track_index))

        # Note that the first track will get id 0, the second id 1, etc. This is required for the lineages file
        tracks.append(_extract_links_from_track(track_file, graph, min_time_point=min_time_point, max_time_point=max_time_point))

        track_index += 1

    return tracks


def _read_lineage_file(tracks_dir: str, graph: Graph, tracks: List[Track], min_time_point: int = 0,
                       max_time_point: int = 5000) -> None:
    """Connects the lineages in the graph based on information from the lineages.p file"""
    print("Reading lineages file")
    lineage_file = os.path.join(tracks_dir, "lineages.p")
    with open(lineage_file, "rb") as file_handle:
        lineages = pickle.load(file_handle, encoding='latin1')
        for lineage in lineages:
            mother_track = tracks[lineage[0]]
            child_track_1 = tracks[lineage[1]]
            child_track_2 = tracks[lineage[2]]

            first_time_point_after_division = numpy.amin(child_track_1.t)
            if first_time_point_after_division - 1 < min_time_point or first_time_point_after_division > max_time_point:
                continue

            mother_last_snapshot = _get_cell_in_time_point(mother_track, first_time_point_after_division - 1)
            child_1_first_snapshot = _get_cell_in_time_point(child_track_1, first_time_point_after_division)
            child_2_first_snapshot = _get_cell_in_time_point(child_track_2, first_time_point_after_division)

            graph.add_edge(mother_last_snapshot, child_1_first_snapshot)
            graph.add_edge(mother_last_snapshot, child_2_first_snapshot)


def _get_cell_in_time_point(track: Track, time_point_number: int) -> Particle:
    position = track.get_pos(time_point_number)
    particle = Particle(position[0], position[1], position[2])
    particle.with_time_point_number(time_point_number)
    return particle


def _extract_links_from_track(track_file: str, graph: Graph, min_time_point: int = 0, max_time_point: int = 5000) -> Track:
    with open(track_file, "rb") as file_handle:
        track = pickle.load(file_handle, encoding='latin1')
        current_particle = None

        for time_point in track.t:
            if time_point < min_time_point or time_point > max_time_point:
                continue

            previous_particle = current_particle
            current_particle = _get_cell_in_time_point(track, time_point)
            graph.add_node(current_particle)

            if previous_particle is not None:
                graph.add_edge(previous_particle, current_particle)

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
