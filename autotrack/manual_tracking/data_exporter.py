import os.path
import pickle
import sys
from typing import List

import numpy
from networkx import Graph

from autotrack.core import UserError
from autotrack.core.particles import Particle
from autotrack.linking import existing_connections
from autotrack.linking_analysis.cell_appearance_finder import find_appeared_cells


def export_links(links: Graph, folder: str):
    """Exports the links of the experiment in Guizela's file format."""
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.isdir(folder):
        raise UserError("Output folder is not a directory",
                        "Cannot output anything - output folder is not a directory.")
    _TrackExporter(links, folder).export_tracks()


class _TrackExporter:

    _next_track_id: int = 0
    _graph: Graph
    _output_folder: str

    _mother_daughter_pairs: List[List[int]]

    def __init__(self, graph: Graph, output_folder: str):
        self._graph = graph
        self._output_folder = output_folder
        self._mother_daughter_pairs = []

    def _export_track(self, particle: Particle, track_id: int):
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
                self._export_track(daughter_1, track_id_1)
                self._export_track(daughter_2, track_id_2)
                break
            if len(future_particles) == 0:
                break  # End of track

            # Add point to track
            particle = future_particles.pop()
            track.add_point(x=numpy.array([particle.x, particle.y, particle.z]), t=particle.time_point_number())

        with open(os.path.join(self._output_folder, f"track_{track_id:05d}.p"), "wb") as handle:
            pickle.dump(track, handle)

    def export_tracks(self):
        """Exports all tracks to an existing directory. Existing files are overwritten without warning."""
        for particle in find_appeared_cells(self._graph):
            self._export_track(particle, self._get_new_track_id())
        with open(os.path.join(self._output_folder, "lineages.p"), "wb") as handle:
            pickle.dump(self._mother_daughter_pairs, handle)

        self._next_track_id = 0
        self._mother_daughter_pairs = []

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
