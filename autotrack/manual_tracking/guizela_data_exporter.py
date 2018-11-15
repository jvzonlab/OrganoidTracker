import glob
import os.path
import pickle
import re
import sys
from typing import List, Any, Optional, Dict

import numpy
from networkx import Graph

from autotrack.core import UserError
from autotrack.core.particles import Particle
from autotrack.linking import existing_connections
from autotrack.linking_analysis.cell_appearance_finder import find_appeared_cells


def export_links(links: Graph, output_folder: str, comparison_folder: Optional[str] = None):
    """Exports the links of the experiment in Guizela's file format."""
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.isdir(output_folder):
        raise UserError("Output folder is not actually a folder",
                        "Cannot output anything - output folder is not actually a folder. Did you select a file?")
    if comparison_folder is not None and not os.path.isdir(comparison_folder):
        raise UserError("Output folder is not actually a folder",
                        "Cannot fix track ids - comparison folder does not exist.")
    exporter = _TrackExporter(links)
    if comparison_folder is not None:
        exporter.synchronize_ids_with_folder(comparison_folder)
    exporter.export_tracks(output_folder)

class _TrackExporter:
    """Used to export tracks in Guizela's file format, preferably using the original track ids."""

    _next_track_id: int = 0
    _graph: Graph

    _mother_daughter_pairs: List[List[int]]
    _tracks_by_id: Dict[int, Any]

    def __init__(self, graph: Graph):
        self._graph = graph
        self._mother_daughter_pairs = []
        self._tracks_by_id = {}

        # Convert graph to list of tracks
        for particle in find_appeared_cells(self._graph):
            self._add_track_including_child_tracks(particle, self._get_new_track_id())

    def _add_track_including_child_tracks(self, particle: Particle, track_id: int):
        """Exports a track, starting from the first particle position in the track."""
        _allow_loading_classes_without_namespace()
        import track_lib_v4

        track = track_lib_v4.Track(x=numpy.array([particle.x, particle.y, particle.z]), t=particle.time_point_number())

        while True:
            future_particles = existing_connections.find_future_particles(self._graph, particle)
            if len(future_particles) == 2:
                # Division: end current track and start two new ones
                daughter_1 = future_particles.pop()
                daughter_2 = future_particles.pop()
                track_id_1 = self._get_new_track_id()
                track_id_2 = self._get_new_track_id()
                self._mother_daughter_pairs.append([track_id, track_id_1, track_id_2])
                self._add_track_including_child_tracks(daughter_1, track_id_1)
                self._add_track_including_child_tracks(daughter_2, track_id_2)
                break
            if len(future_particles) == 0:
                break  # End of track

            # Add point to track
            particle = future_particles.pop()
            track.add_point(x=numpy.array([particle.x, particle.y, particle.z]), t=particle.time_point_number())

        self._tracks_by_id[track_id] = track

    def synchronize_ids_with_folder(self, input_folder: str):
        for file in os.listdir(input_folder):
            track_id_matches = re.findall(r"track_([0-9]+)\.p$", file)
            if len(track_id_matches) != 1:
                continue
            track_id = int(track_id_matches[0])
            with open(os.path.join(input_folder, file), "rb") as handle:
                track = pickle.load(handle, encoding="latin-1")
                first_time_point_number = track.t[0]
                first_xyz = track.x[0]
                if str(first_xyz) == "[313.33617459 303.61768616  14.        ]":
                    print(f"Searching for track start at {first_xyz} at t={first_time_point_number}")
                matching_track_id = self._find_track_id_beginning_at(first_time_point_number, *first_xyz)
                if matching_track_id is not None:
                    self._swap_ids(track_id, matching_track_id)

    def _find_track_id_beginning_at(self, time_point_number: int, x: float, y: float, z: float) -> Optional[int]:
        nearest_track_distance_squared = None
        nearest_track_id = None

        for track_id, track in self._tracks_by_id.items():
            if track.t[0] != time_point_number:
                continue  # At another time point, ignore
            xyz = track.x[0]
            distance_squared = (x - xyz[0]) ** 2 + (y - xyz[1]) ** 2 + ((z - xyz[2]) * 6) ** 2
            if nearest_track_distance_squared is None or nearest_track_distance_squared > distance_squared:
                nearest_track_id = track_id
                nearest_track_distance_squared = distance_squared

        if nearest_track_distance_squared is None or nearest_track_distance_squared > 1000:
            return None
        return nearest_track_id

    def export_tracks(self, output_folder: str):
        """Exports all tracks to an existing directory. Existing files are overwritten without warning."""
        for track_id, track in self._tracks_by_id.items():
            if track is not None:
                with open(os.path.join(output_folder, f"track_{track_id:05d}.p"), "wb") as handle:
                    pickle.dump(track, handle)
        with open(os.path.join(output_folder, "lineages.p"), "wb") as handle:
            pickle.dump(self._mother_daughter_pairs, handle)

    def _swap_ids(self, id1: int, id2: int):
        """All tracks with id1 will have id2, and vice versa."""
        # Swap ids in mother/daughter pairs
        for mother_daughter_pair in self._mother_daughter_pairs:
            for i in range(len(mother_daughter_pair)):
                if mother_daughter_pair[i] == id1:
                    mother_daughter_pair[i] = id2
                elif mother_daughter_pair[i] == id2:
                    mother_daughter_pair[i] = id1

        # Swap ids in track dictionary
        new_track_2 = self._tracks_by_id.get(id1)
        new_track_1 = self._tracks_by_id.get(id2)
        self._tracks_by_id[id1] = new_track_1
        self._tracks_by_id[id2] = new_track_2

    def _get_new_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id


def _allow_loading_classes_without_namespace():
    """Inserts the current directory once to the Python module path. This makes Python able to find the necessary
    modules without requiring the "autotrack.manual_tracking" namespace. If we would include that namespace, then
    Guizela's software wouldn't be able to read our files."""
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.index(path)
    except ValueError:
        print("Adding to Python path: " + path)
        sys.path.insert(0, path)
