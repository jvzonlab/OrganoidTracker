import os.path
import pickle
import re
import sys
from typing import List, Any, Optional, Dict

import numpy

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import ImageOffsets
from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.guizela_tracker_compatibility import cell_type_converter
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.position_analysis import position_markers


def export_links(experiment: Experiment, output_folder: str, comparison_folder: Optional[str] = None):
    """Exports the links of the experiment in Guizela's file format."""
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.isdir(output_folder):
        raise UserError("Output folder is not actually a folder",
                        "Cannot output anything - output folder is not actually a folder. Did you select a file?")
    if comparison_folder is not None and not os.path.isdir(comparison_folder):
        raise UserError("Output folder is not actually a folder",
                        "Cannot fix track ids - comparison folder does not exist.")
    exporter = _TrackExporter(experiment)
    if comparison_folder is not None:
        exporter.synchronize_ids_with_folder(comparison_folder)
    exporter.export_tracks(output_folder)


class _TrackExporter:
    """Used to export tracks in Guizela's file format, preferably using the original track ids."""

    _next_track_id: int = 0
    _links: Links
    _positions: PositionCollection
    _offsets: ImageOffsets

    _mother_daughter_pairs: List[List[int]]
    _dead_track_ids: List[int]
    _typed_track_ids: Dict[str, List[int]]
    _tracks_by_id: Dict[int, Any]

    def __init__(self, experiment: Experiment):
        self._links = experiment.links
        self._positions = experiment.positions
        self._offsets = experiment.images.offsets
        self._mother_daughter_pairs = []
        self._dead_track_ids = []
        self._typed_track_ids = dict()
        self._tracks_by_id = {}

        # Convert graph to list of tracks
        self._links.sort_tracks_by_x()
        for track_id, track in self._links.find_all_tracks_and_ids():
            self._add_track_including_child_tracks(track, track_id)

    def _add_track_including_child_tracks(self, track: LinkingTrack, track_id: int):
        """Exports a track, starting from the first position position in the track."""
        _allow_loading_classes_without_namespace()
        import track_lib_v4

        guizela_track = None
        for position in track.positions():
            moved_position = position - self._offsets.of_time_point(position.time_point())
            if guizela_track is None:
                guizela_track = track_lib_v4.Track(x=numpy.array([moved_position.x, moved_position.y, moved_position.z]),
                                                   t=position.time_point_number())
            else:
                guizela_track.add_point(x=numpy.array([moved_position.x, moved_position.y, moved_position.z]),
                                        t=position.time_point_number())

        # Register cell type
        position_type = position_markers.get_position_type(self._positions, track.find_last_position())
        if position_type is not None:
            track_ids_of_cell_type = self._typed_track_ids.get(position_type)
            if track_ids_of_cell_type is None:
                # Start new list
                self._typed_track_ids[position_type] = [track_id]
            else:
                # Append to existing list
                track_ids_of_cell_type.append(track_id)

        future_tracks = track.get_next_tracks()
        if len(future_tracks) == 2:
            # Division: end current track and start two new ones
            daughter_track_1, daughter_track_2 = future_tracks
            track_id_1 = self._links.get_track_id(daughter_track_1)
            track_id_2 = self._links.get_track_id(daughter_track_2)
            self._mother_daughter_pairs.append([track_id, track_id_1, track_id_2])
            self._add_track_including_child_tracks(daughter_track_1, track_id_1)
            self._add_track_including_child_tracks(daughter_track_2, track_id_2)
        elif len(future_tracks) == 0:
            # End of the road
            if not linking_markers.is_live(self._positions, track.find_last_position()):
                # Actual cell dead, mark as such
                self._dead_track_ids.append(track_id)

        self._tracks_by_id[track_id] = guizela_track

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
                    pickle.dump(track, handle, protocol=2)
        with open(os.path.join(output_folder, "lineages.p"), "wb") as handle:
            pickle.dump(self._mother_daughter_pairs, handle, protocol=2)
        with open(os.path.join(output_folder, "dead_cells.p"), "wb") as handle:
            pickle.dump(self._dead_track_ids, handle, protocol=2)
        for cell_type, track_id_list in self._typed_track_ids.items():
            file_name = cell_type_converter.CELL_TYPE_TO_FILE.get(cell_type)
            if file_name is not None:
                with open(os.path.join(output_folder, file_name), "wb") as handle:
                    pickle.dump(track_id_list, handle, protocol=2)

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

        # Swap ids in cell deaths and typed cell lists
        self._swap_in_list(self._dead_track_ids, id1, id2)
        for track_id_list in self._typed_track_ids.values():
            self._swap_in_list(track_id_list, id1, id2)

    def _swap_in_list(self, list: List[int], id1: int, id2: int):
        id1_affected = id1 in list
        id2_affected = id2 in list
        if id1_affected and not id2_affected:
            # Swap id1 with id2
            list.remove(id1)
            list.append(id2)
        elif not id1_affected and id2_affected:
            # Swap id2 with id1
            list.remove(id2)
            list.append(id1)
        else:
            pass  # Both or none were affected - no need to swap


def _allow_loading_classes_without_namespace():
    """Inserts the current directory once to the Python module path. This makes Python able to find the necessary
    modules without requiring the "organoid_tracker.manual_tracking" namespace. If we would include that namespace, then
    Guizela's software wouldn't be able to read our files."""
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.index(path)
    except ValueError:
        print("Adding to Python path: " + path)
        sys.path.insert(0, path)
