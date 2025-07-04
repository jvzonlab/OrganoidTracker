"""Classes for expressing the positions of positions"""
import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

import numpy

from organoid_tracker.core import TimePoint, UserError, Color, image_coloring
from organoid_tracker.core.beacon_collection import BeaconCollection
from organoid_tracker.core.connections import Connections
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.global_data import GlobalData
from organoid_tracker.core.image_filters import ImageFilters
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import ImageOffsets, ChannelDescription
from organoid_tracker.core.link_data import LinkData
from organoid_tracker.core.links import Links, LinkingTrack
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution, ImageTimings
from organoid_tracker.core.spline import SplineCollection, Spline
from organoid_tracker.core.warning_limits import WarningLimits
from organoid_tracker.image_loading.builtin_image_filters import ThresholdFilter, GaussianBlurFilter, \
    MultiplyPixelsFilter, InterpolatedMinMaxFilter, IntensityPoint
from organoid_tracker.linking_analysis import linking_markers

FILE_EXTENSION = "aut"
SUPPORTED_IMPORT_FILES = [
    (FILE_EXTENSION.upper() + " file", "*." + FILE_EXTENSION),
    ("Detection or linking files", "*.json"),
    ("Cell tracking challenge files", "*.txt"),
    ("TrackMate file", "*.xml"),
    ("Guizela's tracking files", "track_00000.p")]
WRITE_NEW_FORMAT = True  # Default value for the saving function. Reading always supports both formats.


def load_positions_and_shapes_from_json(experiment: Experiment, json_file_name: str,
                                        min_time_point: int = 0, max_time_point: int = 5000):
    """Loads a JSON file that contains position positions, with or without shape information."""
    time_points = _read_json_from_file(json_file_name)
    _parse_simple_position_format(experiment, time_points, min_time_point, max_time_point)


def _load_guizela_data_file(experiment: Experiment, file_name: str, min_time_point: int, max_time_point: int):
    """Starting from a random *.p file in a directory, this loads all data according to Guizela's format from that
    directory."""
    from organoid_tracker.guizela_tracker_compatibility import guizela_data_importer
    guizela_data_importer.add_data_to_experiment(experiment, os.path.dirname(file_name), min_time_point, max_time_point)


def _load_cell_tracking_challenge_file(experiment: Experiment, file_name: str, min_time_point: int,
                                       max_time_point: int):
    from organoid_tracker.imaging import ctc_io
    ctc_io.load_data_file(file_name, min_time_point, max_time_point, experiment=experiment)


def _load_trackmate_file(experiment: Experiment, file_name: str, min_time_point: int, max_time_point: int):
    from organoid_tracker.imaging import trackmate_io
    trackmate_io.load_data_file(file_name, min_time_point, max_time_point, experiment=experiment)


def load_data_file(file_name: str, min_time_point: int = 0, max_time_point: int = 5000, *,
                   experiment: Optional[Experiment] = None) -> Experiment:
    """Loads some kind of data file. This should support all data formats of our research group. Raises ValueError if
    the file fails to load. All data is dumped into the given experiment object."""
    if experiment is None:
        experiment = Experiment()

    if file_name.lower().endswith("." + FILE_EXTENSION) or file_name.lower().endswith(".json"):
        _load_json_data_file(experiment, file_name, min_time_point, max_time_point)
        return experiment
    elif file_name.lower().endswith(".p"):
        _load_guizela_data_file(experiment, file_name, min_time_point, max_time_point)
        return experiment
    elif file_name.lower().endswith(".txt"):
        _load_cell_tracking_challenge_file(experiment, file_name, min_time_point, max_time_point)
        return experiment
    elif file_name.lower().endswith(".xml"):
        _load_trackmate_file(experiment, file_name, min_time_point, max_time_point)
        return experiment
    else:
        raise ValueError(f"Cannot load data from file \"{file_name}\": it is of an unknown format")


def _parse_timings(data: Dict[str, Any], min_time_point: int, max_time_point: int):
    min_time_point_number = data["min_time_point"]
    timings = data["timings_m"]
    return ImageTimings(min_time_point_number, numpy.array(timings, dtype=numpy.float64)) \
        .limit_to_time(min_time_point, max_time_point)


def _parse_channel_description(data: Dict[str, Any]) -> ChannelDescription:
    colormap = image_coloring.get_colormap(data["colormap"])
    return ChannelDescription(channel_name=data["name"], colormap=colormap)


def _load_json_data_file(experiment: Experiment, file_name: str, min_time_point: int, max_time_point: int):
    """Loads any kind of JSON file."""
    data = _read_json_from_file(file_name)

    # Let the experiment overwrite this file upon the next save
    experiment.last_save_file = file_name

    if "version" not in data and "family_scores" not in data:
        # We don't have a general data file, but a specialized one
        if "directed" in data:
            # File is a linking result file
            _parse_d3_links_format(experiment, data, min_time_point, max_time_point)
        else:  # file is a position/shape file
            _parse_simple_position_format(experiment, data, min_time_point, max_time_point)
        return experiment

    version = data.get("version", "v1")

    if version == "v1":
        _load_json_data_file_v1(experiment, data, min_time_point, max_time_point)
        return

    if version == "v2":
        _load_json_data_file_v2(experiment, data, min_time_point, max_time_point)
        return

    raise UserError("Unknown data version",
                    f"The version of this program is not able to load data of version {version}."
                    f" Maybe your version is outdated?")


def _load_json_data_file_v2(experiment: Experiment, data: Dict[str, Any], min_time_point: int, max_time_point: int):
    """Loads any kind of JSON file."""

    if "name" in data:
        is_automatic = bool(data.get("name_is_automatic"))
        experiment.name.set_name(data["name"], is_automatic=is_automatic)

    if "positions" in data:
        _parse_positions_and_meta_format(experiment, data["positions"], min_time_point, max_time_point)

    if "tracks" in data:
        _parse_tracks_and_meta_format(experiment, data["tracks"], min_time_point, max_time_point)

    if "data_axes" in data:
        _parse_splines_format(experiment, data["data_axes"], min_time_point, max_time_point)

    if "data_axes_meta" in data:
        _parse_splines_meta_format(experiment, data["data_axes_meta"])

    if "beacons" in data:
        _parse_beacons_format(experiment, data["beacons"], min_time_point, max_time_point)

    if "connections" in data:
        _parse_connections_format(experiment, data["connections"], min_time_point, max_time_point)

    if "warning_limits" in data:
        experiment.warning_limits = WarningLimits(**data["warning_limits"])

    if "image_resolution" in data:
        x_res = data["image_resolution"]["x_um"]
        y_res = data["image_resolution"]["y_um"]
        z_res = data["image_resolution"]["z_um"]
        t_res = data["image_resolution"]["t_m"]
        experiment.images.set_resolution(ImageResolution(x_res, y_res, z_res, t_res))

    if "image_offsets" in data:
        experiment.images.offsets = ImageOffsets([_parse_position(entry) for entry in data["image_offsets"]])

    if "image_filters" in data:
        experiment.images.filters = _parse_image_filters(data["image_filters"])

    if "image_timings" in data:
        experiment.images.set_timings(_parse_timings(data["image_timings"], min_time_point, max_time_point))

    if "image_channel_descriptions" in data:
        for channel_index_zero, channel_json in enumerate(data["image_channel_descriptions"]):
            experiment.images.set_channel_description(ImageChannel(index_zero=channel_index_zero),
                                                      _parse_channel_description(channel_json))

    if "color" in data:
        color = data["color"]
        experiment.color = Color.from_rgb_floats(color[0], color[1], color[2])

    if "other_data" in data:
        experiment.global_data = GlobalData(data["other_data"])


def _load_json_data_file_v1(experiment: Experiment, data: Dict[str, Any], min_time_point: int, max_time_point: int):
    """Loads any kind of JSON file."""

    if "name" in data:
        is_automatic = bool(data.get("name_is_automatic"))
        experiment.name.set_name(data["name"], is_automatic=is_automatic)

    if "shapes" in data:
        # Deprecated, nowadays stored in "positions"
        _parse_simple_position_format(experiment, data["shapes"], min_time_point, max_time_point)
    elif "positions" in data:
        _parse_simple_position_format(experiment, data["positions"], min_time_point, max_time_point)

    if "data_axes" in data:
        _parse_splines_format(experiment, data["data_axes"], min_time_point, max_time_point)

    if "data_axes_meta" in data:
        _parse_splines_meta_format(experiment, data["data_axes_meta"])

    if "beacons" in data:
        _parse_beacons_format(experiment, data["beacons"], min_time_point, max_time_point)

    if "connections" in data:
        _parse_connections_format(experiment, data["connections"], min_time_point, max_time_point)

    if "warning_limits" in data:
        experiment.warning_limits = WarningLimits(**data["warning_limits"])

    if "links" in data:
        _parse_d3_links_format(experiment, data["links"], min_time_point, max_time_point)
    elif "links_scratch" in data:  # Deprecated, was used back when experiments could hold multiple linking sets
        _parse_d3_links_format(experiment, data["links_scratch"], min_time_point, max_time_point)
    elif "links_baseline" in data:  # Deprecated, was used back when experiments could hold multiple linking sets
        _parse_d3_links_format(experiment, data["links_baseline"], min_time_point, max_time_point)

    if "image_resolution" in data:
        x_res = data["image_resolution"]["x_um"]
        y_res = data["image_resolution"]["y_um"]
        z_res = data["image_resolution"]["z_um"]
        t_res = data["image_resolution"]["t_m"]
        experiment.images.set_resolution(ImageResolution(x_res, y_res, z_res, t_res))

    if "image_offsets" in data:
        experiment.images.offsets = ImageOffsets([_parse_position(entry) for entry in data["image_offsets"]])

    if "image_filters" in data:
        experiment.images.filters = _parse_image_filters(data["image_filters"])

    if "image_timings" in data:
        experiment.images.set_timings(_parse_timings(data["image_timings"], min_time_point, max_time_point))

    if "image_channel_descriptions" in data:
        for channel_index_zero, channel_json in enumerate(data["image_channel_descriptions"]):
            experiment.images.set_channel_description(ImageChannel(index_zero=channel_index_zero),
                                                      _parse_channel_description(channel_json))

    if "color" in data:
        color = data["color"]
        experiment.color = Color.from_rgb_floats(color[0], color[1], color[2])

    if "other_data" in data:
        experiment.global_data = GlobalData(data["other_data"])


def _parse_simple_position_format(experiment: Experiment, json_structure: Dict[str, List], min_time_point: int,
                                  max_time_point: int):
    positions = experiment.positions

    for time_point_number, raw_positions in json_structure.items():
        time_point_number = int(time_point_number)  # str -> int
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue

        for raw_position in raw_positions:
            position = Position(*raw_position[0:3], time_point_number=time_point_number)
            positions.add(position)


def _parse_d3_links_format(experiment: Experiment, links_json: Dict[str, Any], min_time_point: int,
                           max_time_point: int):
    """Parses a node_link_graph and adds all links and positions to the experiment."""
    links = experiment.links
    position_data = experiment.position_data
    link_data = experiment.link_data
    _add_d3_data(links, link_data, position_data, links_json, min_time_point, max_time_point)
    positions = experiment.positions
    for position in links.find_all_positions():
        positions.add(position)


def _parse_positions_and_meta_format(experiment: Experiment, positions_json: List[Dict], min_time_point: int,
                                     max_time_point: int):
    positions = experiment.positions

    for time_point_json in positions_json:
        time_point_number = time_point_json["time_point"]
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue

        has_meta = "position_meta" in time_point_json
        positions_of_time_point = list() if has_meta else None
        for raw_position in time_point_json["coords_xyz_px"]:
            position = Position(*raw_position, time_point_number=time_point_number)
            positions.add(position)
            if positions_of_time_point is not None:
                positions_of_time_point.append(position)

        if has_meta:
            experiment.position_data.add_data_from_time_point_dict(TimePoint(time_point_number), positions_of_time_point,
                                                                   time_point_json["position_meta"])


def _parse_tracks_and_meta_format(experiment: Experiment, tracks_json: List[Dict], min_time_point: int,
                                  max_time_point: int):
    links = experiment.links
    link_data = experiment.link_data

    # Iterate a first time to add the tracks
    for track_json in tracks_json:
        time_point_number_start = track_json["time_point_start"]
        time_point_number_end = time_point_number_start + len(track_json["coords_xyz_px"]) - 1
        if time_point_number_end < min_time_point or time_point_number_start > max_time_point:
            continue  # Can skip this track entirely

        coords_xyz_px = track_json["coords_xyz_px"]
        positions_of_track = list()
        min_index = max(0, min_time_point - time_point_number_start)
        max_index = min(len(coords_xyz_px) - 1, max_time_point - time_point_number_start)
        for i in range(min_index, max_index + 1):
            position = Position(*coords_xyz_px[i], time_point_number=time_point_number_start + i)
            positions_of_track.append(position)
        track = LinkingTrack(positions_of_track)
        links.add_track(track)

        # Handle link metadata
        if "link_meta" in track_json:
            for metadata_key, metadata_values in track_json["link_meta"].items():
                for i in range(min_index, min(max_index, len(metadata_values))):
                    value = metadata_values[i]
                    if value is None:
                        continue

                    link_data.set_link_data(positions_of_track[i - min_index], positions_of_track[i - min_index + 1], metadata_key, value)

        # Handle lineage metadata
        if "lineage_meta" in track_json:
            for metadata_key, metadata_value in track_json["lineage_meta"].items():
                links.set_lineage_data(track, metadata_key, metadata_value)

    # Iterate again to add connections to previous tracks
    for track_json in tracks_json:
        if "coords_xyz_px_before" not in track_json:
            continue
        time_point_number_start = track_json["time_point_start"]
        if time_point_number_start <= min_time_point:
            continue  # Ignore tracks that start at/after the minimum time point - can't add metadata to before
        if time_point_number_start > max_time_point:
            continue  # Also ignore tracks starting after the maximum time point - the time point before doesn't exist

        position_first = Position(*track_json["coords_xyz_px"][0], time_point_number=time_point_number_start)
        metadata = track_json.get("link_meta_before")
        for i, raw_position in enumerate(track_json["coords_xyz_px_before"]):
            # Connect the tracks
            position_previous_track = Position(*raw_position, time_point_number=time_point_number_start - 1)
            previous_track = links.get_track(position_previous_track)
            current_track = links.get_track(position_first)
            links.connect_tracks(previous=previous_track, next=current_track)

            # And add metadata for those links
            if metadata is not None:
                for metadata_key, metadata_values in metadata.items():
                    if i >= len(metadata_values):
                        continue
                    value = metadata_values[i]
                    if value is None:
                        continue
                    link_data.set_link_data(position_previous_track, position_first, metadata_key, value)


def _parse_splines_format(experiment: Experiment, splines_data: List[Dict], min_time_point: int, max_time_point: int):
    for spline_json in splines_data:
        time_point_number = spline_json["_time_point_number"]
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue
        spline = Spline()
        points_x = spline_json["x_list"]
        points_y = spline_json["y_list"]
        if "z_list" in spline_json:
            points_z = spline_json["z_list"]
        else:
            points_z = [spline_json["z"]] * len(points_y)  # Old format, when axes where 2D
        for i in range(len(points_x)):
            spline.add_point(points_x[i], points_y[i], points_z[i])
        spline.set_offset(spline_json["offset"])
        spline_id = int(spline_json["id"]) if "id" in spline_json else None
        experiment.splines.add_spline(TimePoint(time_point_number), spline, spline_id)


def _parse_splines_meta_format(experiment: Experiment, axes_meta: Dict[str, object]):
    """Currently parses the type of each axis that was drawn."""
    for key, value in axes_meta.items():
        if key == "reference_time_point_number" and isinstance(value, int):
            experiment.splines.reference_time_point(TimePoint(value))
            continue

        try:
            spline_id = int(key)
        except ValueError:
            continue  # Ignore unknown keys
        else:
            # Currently, the only supported keys are "marker" and "is_axis"
            if isinstance(value, dict) and "marker" in value and "is_axis" in value:
                marker = str(value["marker"])
                is_axis = bool(value["is_axis"])
                experiment.splines.set_marker_name(spline_id, marker, is_axis)


def _parse_beacons_format(experiment: Experiment, beacons_data: Dict[str, List], min_time_point: int,
                          max_time_point: int):
    """Expects a dict: `{"1": [...], "2": [...]}`. Keys are time points, values are lists with [x,y,z(,type)] positions:
    `[[2,4,7,"TYPE_NAME"], [4,5.3,3], ...]`. The type is optional"""
    for time_point_str, beacons_list in beacons_data.items():
        time_point_number = int(time_point_str)
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue
        for beacon_values in beacons_list:
            type_name = beacon_values[3] if len(beacon_values) > 3 else None
            experiment.beacons.add(Position(*beacon_values[0:3], time_point_number=time_point_number), type_name)


def _parse_connections_format(experiment: Experiment, connections_data: Dict[str, Dict],
                              min_time_point: int, max_time_point: int):
    """Adds all connections from the serialized format to the Connections object."""
    connections = experiment.connections
    for time_point_str, connections_of_time_point in connections_data.items():
        time_point_number = int(time_point_str)
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue

        if isinstance(connections_of_time_point, list):
            # Old format, without metadata, just a list of tuples
            for connection in connections_of_time_point:
                position1 = _parse_position(connection[0])
                position2 = _parse_position(connection[1])

                connections.add_connection(position1.with_time_point_number(time_point_number),
                                           position2.with_time_point_number(time_point_number))
        else:
            # Dictionary, with keys "from_xyz_px", "to_xyz_px" and "metadata"

            # Parse the positions
            connections_list = []
            for raw_position_1, raw_position_2 in zip(connections_of_time_point["from_xyz_px"], connections_of_time_point["to_xyz_px"]):
                position1 = Position(*raw_position_1, time_point_number=time_point_number)
                position2 = Position(*raw_position_2, time_point_number=time_point_number)
                connections_list.append((position1, position2))

            # Add all connections and metadata
            connections.add_data_from_time_point_dict(TimePoint(time_point_number), connections_list, connections_of_time_point["metadata"])


def _parse_image_filters(data: Dict[str, Any]) -> ImageFilters:
    filters = ImageFilters()
    for key, filters_dict in data.items():
        channel_index = int(key.replace("channel_", ""))
        channel = ImageChannel(index_zero=channel_index)

        for filter_dict in filters_dict:
            if filter_dict["type"] == "threshold":
                filters.add_filter(channel, ThresholdFilter(filter_dict["min"]))
            elif filter_dict["type"] == "gaussian":
                filters.add_filter(channel, GaussianBlurFilter(filter_dict["radius_px"]))
            elif filter_dict["type"] == "multiply":
                filters.add_filter(channel, MultiplyPixelsFilter(filter_dict["factor"]))
            elif filter_dict["type"] == "interpolated_min_max":
                points = dict()
                for point_dict in filter_dict["points"]:
                    points[IntensityPoint(time_point=TimePoint(point_dict["t"]), z=point_dict["z"])] \
                        = point_dict["min"], point_dict["max"]
                filters.add_filter(channel, InterpolatedMinMaxFilter(points))
            else:
                raise ValueError("Unknown image filter: " + filter_dict["type"])
    return filters


def _parse_position(json_structure: Dict[str, Any]) -> Position:
    if "_time_point_number" in json_structure:
        return Position(json_structure["x"], json_structure["y"], json_structure["z"],
                        time_point_number=json_structure["_time_point_number"])
    return Position(json_structure["x"], json_structure["y"], json_structure["z"])


def _add_d3_data(links: Links, link_data: LinkData, position_data: PositionData, links_json: Dict,
                 min_time_point: int = -100000, max_time_point: int = 100000):
    """Adds data in the D3.js node-link format. Used for deserialization."""

    # Add position data
    for node in links_json["nodes"]:
        if len(node.keys()) == 1:
            # No extra data found
            continue
        position = _parse_position(node["id"])
        if position.time_point_number() < min_time_point or position.time_point_number() > max_time_point:
            continue  # Out of range

        for data_key, data_value in node.items():
            if data_key == "id":
                continue

            position_data.set_position_data(position, data_key, data_value)

    # Add links (and link and lineage data)
    for link in links_json["links"]:
        source = _parse_position(link["source"])
        target = _parse_position(link["target"])
        if source.time_point_number() < min_time_point or target.time_point_number() < min_time_point \
                or source.time_point_number() > max_time_point or target.time_point_number() > max_time_point:
            continue  # Ignore time points out of range
        links.add_link(source, target)

        # Now that we have a link, we can add link and lineage data
        for data_key, data_value in link.items():
            if data_key.startswith("__lineage_"):
                # Lineage metadata, store it
                links.set_lineage_data(links.get_track(source), data_key[len("__lineage_"):], data_value)
            elif data_key != "source" and data_key != "target":
                # Link metadata, store it
                link_data.set_link_data(source, target, data_key, data_value)


def _links_to_d3_data(links: Links, positions: Iterable[Position], position_data: PositionData,
                      link_data: LinkData) -> Dict:
    """Return data in D3.js node-link format that is suitable for JSON serialization
    and use in Javascript documents."""
    links.sort_tracks_by_x()  # Make sure tracks are always saved in the correct order

    nodes = list()

    # Save nodes and store extra data
    for position in positions:
        node = {
            "id": _encode_position(position)
        }
        for data_name, data_value in position_data.find_all_data_of_position(position):
            if data_name == "shape":
                continue  # For historical reasons, shape information is stored in the "positions" array
            node[data_name] = data_value
        nodes.append(node)

    # Save edges
    lineage_starting_positions = {track.find_first_position() for track in links.find_starting_tracks()}
    edges = list()
    for source, target in links.find_all_links():
        edge = {
            "source": _encode_position(source),
            "target": _encode_position(target)
        }
        if source in lineage_starting_positions:
            # Start of a lineage, so add lineage data
            for data_name, data_value in links.find_all_data_of_lineage(links.get_track(source)):
                edge["__lineage_" + data_name] = data_value
        for data_name, data_value in link_data.find_all_data_of_link(source, target):
            edge[data_name] = data_value
        edges.append(edge)

    return {
        "directed": False,
        "multigraph": False,
        "graph": dict(),
        "nodes": nodes,
        "links": edges
    }


def save_positions_to_json(experiment: Experiment, json_file_name: str):
    """Saves a list of positions (pixel coordinate) to disk. Because the offset of the images is not saved, the offset
    is added to all positions. This method is mainly intended to export training data for neural networks, so it will
    ignore dead cells."""
    data_structure = _encode_image_positions(experiment)

    _create_parent_directories(json_file_name)

    json_file_name_old = json_file_name + ".OLD"
    if os.path.exists(json_file_name):
        os.rename(json_file_name, json_file_name_old)
    _write_json_to_file(json_file_name, data_structure)
    if os.path.exists(json_file_name_old):
        os.remove(json_file_name_old)


def _encode_image_positions(experiment: Experiment):
    positions = experiment.positions
    offsets = experiment.images.offsets
    position_data = experiment.position_data

    data_structure = {}
    for time_point in positions.time_points():
        offset = offsets.of_time_point(time_point)
        encoded_positions = []
        for position in positions.of_time_point(time_point):
            if linking_markers.is_live(position_data, position):
                encoded_positions.append([position.x - offset.x, position.y - offset.y, position.z - offset.z])

        data_structure[str(time_point.time_point_number())] = encoded_positions
    return data_structure


def _encode_beacons(beacons: BeaconCollection):
    data_structure = {}
    for time_point in beacons.time_points():
        encoded_positions = []
        for beacon in beacons.of_time_point_with_type(time_point):
            if beacon.beacon_type is None:
                encoded_positions.append([beacon.position.x, beacon.position.y, beacon.position.z])
            else:
                encoded_positions.append([beacon.position.x, beacon.position.y, beacon.position.z, beacon.beacon_type])

        data_structure[str(time_point.time_point_number())] = encoded_positions
    return data_structure


def _encode_positions_in_old_format(positions: PositionCollection):
    data_structure = {}
    for time_point in positions.time_points():
        encoded_positions = []
        for position in positions.of_time_point(time_point):
            encoded_positions.append([position.x, position.y, position.z])

        data_structure[str(time_point.time_point_number())] = encoded_positions
    return data_structure


def _encode_data_axes_to_json(data_axes: SplineCollection) -> List[Dict]:
    json_list = list()
    for spline_id, time_point, spline in data_axes.all_splines():
        points_x, points_y, points_z = spline.get_points_3d()
        json_object = {
            "_time_point_number": time_point.time_point_number(),
            "x_list": points_x,
            "y_list": points_y,
            "z_list": points_z,
            "z": spline.get_z(),  # For compatibility with older OrganoidTracker versions (December 2022)
            "offset": spline.get_offset(),
            "id": spline_id
        }
        json_list.append(json_object)
    return json_list


def _encode_data_axes_meta_to_json(splines: SplineCollection) -> Dict[str, Dict[str, str]]:
    json_dict = {}
    reference_time_point = splines.reference_time_point()
    if reference_time_point is not None:
        json_dict["reference_time_point_number"] = reference_time_point.time_point_number()
    for spline_id, marker_name in splines.get_marker_names():
        json_dict[str(spline_id)] = {"marker": marker_name, "is_axis": splines.is_axis(spline_id)}
    return json_dict


def _encode_position(position: Position) -> Dict[str, Any]:
    if position.time_point_number() is None:
        return {
            "x": position.x,
            "y": position.y,
            "z": position.z
        }
    else:
        return {
            "x": position.x,
            "y": position.y,
            "z": position.z,
            "_time_point_number": position.time_point_number()
        }


def _encode_connections_to_json(connections: Connections) -> Dict[str, Dict]:
    connections_dict = dict()

    for time_point in connections.time_points():
        data_names = connections.find_all_data_names(time_point=time_point)

        # Initialize the empty lists for positions and metadata
        positions_1_list = list()
        positions_2_list = list()
        connections_metadata = dict()
        for data_name in data_names:
            connections_metadata[data_name] = list()

        # Loop through all connections at this time point and add them to the lists
        for position_1, position_2, metadata in connections.of_time_point_with_data(time_point):
            positions_1_list.append([position_1.x, position_1.y, position_1.z])
            positions_2_list.append([position_2.x, position_2.y, position_2.z])
            for data_name in data_names:
                connections_metadata[data_name].append(metadata.get(data_name))

        connections_dict[str(time_point.time_point_number())] = {
            "from_xyz_px": positions_1_list,
            "to_xyz_px": positions_2_list,
            "metadata": connections_metadata
        }

    return connections_dict


def _encode_image_filters_to_json(filters: ImageFilters) -> Dict[str, Any]:
    result_dict = dict()
    for image_channel, filters_of_channel in filters.items():
        filter_dicts = list()
        for filter in filters_of_channel:
            if isinstance(filter, ThresholdFilter):
                filter_dicts.append({
                    "type": "threshold",
                    "min": filter.noise_limit
                })
            elif isinstance(filter, GaussianBlurFilter):
                filter_dicts.append({
                    "type": "gaussian",
                    "radius_px": filter.blur_radius
                })
            elif isinstance(filter, MultiplyPixelsFilter):
                filter_dicts.append({
                    "type": "multiply",
                    "factor": filter.factor
                })
            elif isinstance(filter, InterpolatedMinMaxFilter):
                filter_dicts.append({
                    "type": "interpolated_min_max",
                    "points": [
                        {"min": values[0], "max": values[1], "t": point.time_point.time_point_number(), "z": point.z}
                        for point, values in filter.points.items()]
                })
            else:
                warnings.warn("Unknown filter: " + str(filter.get_name()))
        result_dict[f"channel_{image_channel.index_zero}"] = filter_dicts
    return result_dict


def _encode_positions_and_meta(positions: PositionCollection, position_data: PositionData) -> List[Dict]:
    """Encodes positions and metadata to a JSON structure."""
    time_points_json = list()
    for time_point in positions.time_points():
        metadata_lists = dict()
        positions_of_time_point = list(positions.of_time_point(time_point))
        metadata_lists = position_data.create_time_point_dict(time_point, positions_of_time_point)
        xyz_values = [[position.x, position.y, position.z] for position in positions_of_time_point]

        if len(positions_of_time_point) > 0:
            time_point_json = {
                "time_point": time_point.time_point_number(),
                "coords_xyz_px": xyz_values
            }
            if len(metadata_lists) > 0:
                time_point_json["position_meta"] = metadata_lists
            time_points_json.append(time_point_json)
    return time_points_json


def _encode_tracks_and_meta(links: Links, link_data: LinkData) -> List[Dict]:
    tracks_json = list()
    for track in links.find_all_tracks():
        # Collect last positions of previous tracks, for connecting tracks
        coords_xyz_px_before = list()
        link_meta_before = dict()
        first_position = track.find_first_position()
        previous_tracks = track.get_previous_tracks()
        for previous_track in previous_tracks:
            last_position = previous_track.find_last_position()

            # Add metadata to the metadata lists
            for data_name, data_value in link_data.find_all_data_of_link(last_position, first_position):
                # This data value was not yet found in the connections to the previous tracks
                # Set it to None for all previous positions
                if data_name not in link_meta_before:
                    link_meta_before[data_name] = [None] * len(coords_xyz_px_before)
                link_meta_before[data_name].append(data_value)

            coords_xyz_px_before.append([last_position.x, last_position.y, last_position.z])

        # Make sure all metadata lists are the same length, so append None for missing values
        for value_list in link_meta_before.values():
            if len(value_list) < len(coords_xyz_px_before)+1:
                value_list.append(None)

        # Collect positions of this track
        coords_xyz_px = list()
        # Note that the metadata lists are one shorter than the coords list, since link metadata exists between
        # two positions
        link_meta = dict()
        previous_position = None
        for position in track.positions():
            if previous_position is not None:
                # Add metadata in between current and previous position to the metadata lists
                for data_name, data_value in link_data.find_all_data_of_link(previous_position, position):
                    # This data value was not yet found in the connections to the previous tracks
                    # Set it to None for all previous positions
                    if data_name not in link_meta:
                        link_meta[data_name] = [None] * (len(coords_xyz_px) - 1)
                    link_meta[data_name].append(data_value)

                # Make sure all metadata lists are the same length, so append None for missing values
                for value_list in link_meta.values():
                    if len(value_list) < len(coords_xyz_px): #mistake - 1:
                        value_list.append(None)

            coords_xyz_px.append([position.x, position.y, position.z])
            previous_position = position

        track_json = {
            "time_point_start": track.first_time_point_number(),
            "coords_xyz_px": coords_xyz_px
        }
        if len(link_meta) > 0:
            track_json["link_meta"] = link_meta
        if len(coords_xyz_px_before) > 0:
            # Connections to previous tracks, so add connections and possibly metadata
            track_json["coords_xyz_px_before"] = coords_xyz_px_before
            if len(link_meta_before) > 0:
                track_json["link_meta_before"] = link_meta_before
        if len(previous_tracks) == 0 and len(track._lineage_data) > 0:
            # Start of a lineage, so add lineage metadata
            track_json["lineage_meta"] = track._lineage_data

        tracks_json.append(track_json)
    return tracks_json


def save_data_to_json(experiment: Experiment, json_file_name: str, *, write_new_format: bool = WRITE_NEW_FORMAT):
    """Saves positions, shapes, scores and links to a JSON file. The file should end with the extension FILE_EXTENSION.

    We can save in the old v1 or the newer v2 format, which stores the positions and tracks in a more efficient way.
    See TRACKING_FORMATS.md in the manual for more information.
    """
    # Record where file has been saved to
    experiment.last_save_file = json_file_name

    if write_new_format:
        save_data = {"version": "v2"}

        # Save positions
        if experiment.positions.has_positions():
            save_data["positions"] = _encode_positions_and_meta(experiment.positions, experiment.position_data)

        # Save tracks
        if experiment.links.has_links():
            save_data["tracks"] = _encode_tracks_and_meta(experiment.links, experiment.link_data)
    else:
        save_data = {"version": "v1"}

        # Save positions
        if experiment.positions.has_positions():
            save_data["positions"] = _encode_positions_in_old_format(experiment.positions)

        # Save links
        if experiment.links.has_links() or experiment.position_data.has_position_data():
            save_data["links"] = _links_to_d3_data(experiment.links, experiment.positions, experiment.position_data,
                                                   experiment.link_data)

    # Save name
    if experiment.name.has_name():
        save_data["name"] = str(experiment.name)
        save_data["name_is_automatic"] = experiment.name.is_automatic()

    # Save data axes
    if experiment.splines.has_splines():
        save_data["data_axes"] = _encode_data_axes_to_json(experiment.splines)
        save_data["data_axes_meta"] = _encode_data_axes_meta_to_json(experiment.splines)

    # Save beacons
    if experiment.beacons.has_beacons():
        save_data["beacons"] = _encode_beacons(experiment.beacons)

    # Save warning limits
    if experiment.links.has_links():
        save_data["warning_limits"] = experiment.warning_limits.to_dict()

    # Save connections
    if experiment.connections.has_connections():
        save_data["connections"] = _encode_connections_to_json(experiment.connections)

    # Save image resolution
    try:
        resolution = experiment.images.resolution()
        save_data["image_resolution"] = {"x_um": resolution.pixel_size_zyx_um[2],
                                         "y_um": resolution.pixel_size_zyx_um[1],
                                         "z_um": resolution.pixel_size_zyx_um[0],
                                         "t_m": resolution.time_point_interval_m}
    except UserError:
        pass

    # Save image offsets
    save_data["image_offsets"] = [_encode_position(position) for position in experiment.images.offsets.to_list()]

    # Save image filters
    if experiment.images.filters.has_filters():
        save_data["image_filters"] = _encode_image_filters_to_json(experiment.images.filters)

    # Save image timing
    if experiment.images.has_timings() and not experiment.images.timings().is_simple_multiplication():
        save_data["image_timings"] = {"min_time_point": experiment.images.timings().min_time_point_number(),
                                      "timings_m": list(experiment.images.timings().get_cumulative_timings_array_m())}

    # Save image channel descriptions
    save_data["image_channel_descriptions"] = []
    for image_channel in experiment.images.get_channels():
        channel_description = experiment.images.get_channel_description(image_channel)
        save_data["image_channel_descriptions"].append({"name": channel_description.channel_name,
                                                        "colormap": channel_description.colormap.name})

    # Save color
    save_data["color"] = list(experiment.color.to_rgb_floats())

    # Save other global data
    if experiment.global_data.has_global_data():
        save_data["other_data"] = experiment.global_data.get_all_data()

    _create_parent_directories(json_file_name)
    json_file_name_old = json_file_name + ".OLD"
    if os.path.exists(json_file_name):
        os.rename(json_file_name, json_file_name_old)
    _write_json_to_file(json_file_name, save_data)
    if os.path.exists(json_file_name_old):
        os.remove(json_file_name_old)


def _read_json_from_file(file_name: str) -> Dict[str, Any]:
    try:
        # Faster
        import orjson
        with open(file_name, "rb") as handle:
            return orjson.loads(handle.read())
    except ModuleNotFoundError:
        # Slower, but doesn't need the orjson library
        import json
        with open(file_name, "r", encoding="utf8") as handle:
            return json.load(handle)


def _write_json_to_file(file_name: str, data_structure):
    try:
        # Faster path
        import orjson
        with open(file_name, "wb") as handle:
            handle.write(orjson.dumps(data_structure, option=orjson.OPT_SERIALIZE_NUMPY))
    except ModuleNotFoundError:
        # SLower path, but only relies on Python standard library
        import json
        with open(file_name, 'w', encoding="utf8") as handle:
            json.dump(data_structure, handle)


def _create_parent_directories(file_name: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
