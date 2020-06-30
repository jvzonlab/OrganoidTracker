"""Classes for expressing the positions of positions"""
import json
import os
from json import JSONEncoder
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

import numpy

from organoid_tracker.core import shape, TimePoint, UserError
from organoid_tracker.core.beacon_collection import BeaconCollection
from organoid_tracker.core.connections import Connections
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import ImageOffsets
from organoid_tracker.core.links import Links
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.spline import SplineCollection, Spline
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.score import ScoredFamily, Score, Family
from organoid_tracker.core.warning_limits import WarningLimits
from organoid_tracker.linking_analysis import linking_markers

FILE_EXTENSION = "aut"
SUPPORTED_IMPORT_FILES = [
        (FILE_EXTENSION.upper() + " file", "*." + FILE_EXTENSION),
        ("Detection or linking files", "*.json"),
        ("Cell tracking challenge files", "*.txt"),
        ("TrackMate file", "*.xml"),
        ("Guizela's tracking files", "track_00000.p")]

def load_positions_and_shapes_from_json(experiment: Experiment, json_file_name: str,
                                        min_time_point: int = 0, max_time_point: int = 5000):
    """Loads a JSON file that contains position positions, with or without shape information."""
    with open(json_file_name) as handle:
        time_points = json.load(handle)
        _parse_shape_format(experiment, time_points, min_time_point, max_time_point)


def _load_guizela_data_file(experiment: Experiment, file_name: str, min_time_point: int, max_time_point: int):
    """Starting from a random *.p file in a directory, this loads all data according to Guizela's format from that
    directory."""
    from organoid_tracker.guizela_tracker_compatibility import guizela_data_importer
    guizela_data_importer.add_data_to_experiment(experiment, os.path.dirname(file_name), min_time_point, max_time_point)


def _load_cell_tracking_challenge_file(experiment: Experiment, file_name: str, min_time_point: int, max_time_point: int):
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


def _load_json_data_file(experiment: Experiment, file_name: str, min_time_point: int, max_time_point: int):
    """Loads any kind of JSON file."""
    with open(file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)

        if "version" not in data and "family_scores" not in data:
            # We don't have a general data file, but a specialized one
            if "directed" in data:
                # File is a linking result file
                _parse_links_format(experiment, data, min_time_point, max_time_point)
            else:  # file is a position/shape file
                _parse_shape_format(experiment, data, min_time_point, max_time_point)
            return experiment

        if data.get("version", "v1") != "v1":
            raise ValueError("Unknown data version", "This program is not able to load data of version "
                             + str(data["version"]) + ".")

        if "name" in data:
            experiment.name.set_name(data["name"])

        if "shapes" in data:
            _parse_shape_format(experiment, data["shapes"], min_time_point, max_time_point)
        elif "positions" in data:
            _parse_shape_format(experiment, data["positions"], min_time_point, max_time_point)

        if "data_axes" in data:
            _parse_splines_format(experiment, data["data_axes"], min_time_point, max_time_point)

        if "data_axes_meta" in data:
            _parse_splines_meta_format(experiment, data["data_axes_meta"])

        if "beacons" in data:
            _parse_beacons_format(experiment, data["beacons"], min_time_point, max_time_point)

        if "connections" in data:
            _parse_connections_format(experiment, data["connections"], min_time_point, max_time_point)

        if "family_scores" in data:
            experiment.scores.add_scored_families(data["family_scores"])

        if "warning_limits" in data:
            experiment.warning_limits = WarningLimits(**data["warning_limits"])

        if "links" in data:
            _parse_links_format(experiment, data["links"], min_time_point, max_time_point)
        elif "links_scratch" in data:  # Deprecated, was used back when experiments could hold multiple linking sets
            _parse_links_format(experiment, data["links_scratch"], min_time_point, max_time_point)
        elif "links_baseline" in data:  # Deprecated, was used back when experiments could hold multiple linking sets
            _parse_links_format(experiment, data["links_baseline"], min_time_point, max_time_point)

        if "image_resolution" in data:
            x_res = data["image_resolution"]["x_um"]
            y_res = data["image_resolution"]["y_um"]
            z_res = data["image_resolution"]["z_um"]
            t_res = data["image_resolution"]["t_m"]
            experiment.images.set_resolution(ImageResolution(x_res, y_res, z_res, t_res))

        if "image_offsets" in data:
            experiment.images.offsets = ImageOffsets(data["image_offsets"])


def _parse_shape_format(experiment: Experiment, json_structure: Dict[str, List], min_time_point: int, max_time_point: int):
    positions = experiment.positions
    position_data = experiment.position_data

    for time_point_number, raw_positions in json_structure.items():
        time_point_number = int(time_point_number)  # str -> int
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue

        for raw_position in raw_positions:
            position = Position(*raw_position[0:3], time_point_number=time_point_number)
            position_shape = shape.from_list(raw_position[3:])
            positions.add(position)
            if not position_shape.is_unknown():
                linking_markers.set_shape(position_data, position, position_shape)


def _parse_links_format(experiment: Experiment, link_data: Dict[str, Any], min_time_point: int, max_time_point: int):
    """Parses a node_link_graph and adds all links and positions to the experiment."""
    links = experiment.links
    position_data = experiment.position_data
    _add_d3_data(links, position_data, link_data, min_time_point, max_time_point)
    positions = experiment.positions
    for position in links.find_all_positions():
        positions.add(position)


def _parse_splines_format(experiment: Experiment, splines_data: List[Dict], min_time_point: int, max_time_point: int):
    for spline_json in splines_data:
        time_point_number = spline_json["_time_point_number"]
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue
        spline = Spline()
        points_x = spline_json["x_list"]
        points_y = spline_json["y_list"]
        z = spline_json["z"]
        for i in range(len(points_x)):
            spline.add_point(points_x[i], points_y[i], z)
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


def _parse_beacons_format(experiment: Experiment, beacons_data: Dict[str, List], min_time_point: int, max_time_point: int):
    """Expects a dict: `{"1": [...], "2": [...]}`. Keys are time points, values are lists with [x,y,z] positions:
    `[[2,4,7], [4,5.3,3], ...]`."""
    for time_point_str, beacons_list in beacons_data.items():
        time_point_number = int(time_point_str)
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue
        for beacon_values in beacons_list:
            experiment.beacons.add(Position(*beacon_values, time_point_number=time_point_number))


def _parse_connections_format(experiment: Experiment, connections_data: Dict[str, List[List[Position]]],
                              min_time_point: int, max_time_point: int):
    """Adds all connections from the serialized format to the Connections object."""
    connections = experiment.connections
    for time_point_str, connections_list in connections_data.items():
        time_point_number = int(time_point_str)
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue

        for connection in connections_list:
            position1 = connection[0]
            position2 = connection[1]

            connections.add_connection(position1.with_time_point_number(time_point_number),
                                       position2.with_time_point_number(time_point_number))


def _add_d3_data(links: Links, position_data: PositionData, data: Dict, min_time_point: int = -100000, max_time_point: int = 100000):
    """Adds data in the D3.js node-link format. Used for deserialization."""

    # Add position data
    for node in data["nodes"]:
        if len(node.keys()) == 1:
            # No extra data found
            continue
        position = node["id"]
        for data_key, data_value in node.items():
            if data_key == "id":
                continue

            position_data.set_position_data(position, data_key, data_value)

    # Add links (and lineage data)
    for link in data["links"]:
        source: Position = link["source"]
        target: Position = link["target"]
        if source.time_point_number() < min_time_point or target.time_point_number() < min_time_point \
            or source.time_point_number() > max_time_point or target.time_point_number() > max_time_point:
            continue  # Ignore time points out of range
        links.add_link(source, target)

        # Now that we have a link, we can add lineage data
        for data_key, data_value in link.items():
            if data_key.startswith("__lineage_"):
                links.set_lineage_data(links.get_track(source), data_key[len("__lineage_"):], data_value)


class _MyEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Position):
            if o.time_point_number() is None:
                return {"x": o.x, "y": o.y, "z": o.z}
            return {"x": o.x, "y": o.y, "z": o.z, "_time_point_number": o.time_point_number()}
        if isinstance(o, Score):
            return o.__dict__

        if isinstance(o, ScoredFamily):
            daughters = list(o.family.daughters)
            return {
                "scores": o.score.dict(),
                "mother": o.family.mother,
                "daughter1": daughters[0],
                "daughter2": daughters[1]
            }

        if isinstance(o, numpy.int32):
            return numpy.asscalar(o)

        return JSONEncoder.default(self, o)


def _my_decoder(json_object):
    if 'x' in json_object and 'y' in json_object and 'z' in json_object:
        if '_time_point_number' in json_object:
            return Position(json_object['x'], json_object['y'], json_object['z'],
                            time_point_number=json_object['_time_point_number'])
        else:
            return Position(json_object['x'], json_object['y'], json_object['z'])
    if 'scores' in json_object and 'mother' in json_object and 'daughter1' in json_object:
        mother = json_object["mother"]
        daughter1 = json_object["daughter1"]
        daughter2 = json_object["daughter2"]
        family = Family(mother, daughter1, daughter2)
        score = Score(**json_object["scores"])
        return ScoredFamily(family, score)
    return json_object


def _links_to_d3_data(links: Links, positions: Iterable[Position], position_data: PositionData) -> Dict:
    """Return data in D3.js node-link format that is suitable for JSON serialization
    and use in Javascript documents."""
    nodes = list()

    # Save nodes and store extra data
    for position in positions:
        node = {
            "id": position
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
            "source": source,
            "target": target
        }
        if source in lineage_starting_positions:
            # Start of a lineage, so add lineage data
            for data_name, data_value in links.find_all_data_of_lineage(links.get_track(source)):
                edge["__lineage_" + data_name] = data_value
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
    with open(json_file_name, 'w') as handle:
        json.dump(data_structure, handle, cls=_MyEncoder)
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
        for position in beacons.of_time_point(time_point):
            encoded_positions.append([position.x, position.y, position.z])

        data_structure[str(time_point.time_point_number())] = encoded_positions
    return data_structure


def _encode_positions_and_shapes(positions: PositionCollection, shapes: PositionData):
    data_structure = {}
    for time_point in positions.time_points():
        encoded_positions = []
        for position in positions.of_time_point(time_point):
            shape = linking_markers.get_shape(shapes, position)
            encoded_positions.append([position.x, position.y, position.z] + shape.to_list())

        data_structure[str(time_point.time_point_number())] = encoded_positions
    return data_structure


def _encode_data_axes_to_json(data_axes: SplineCollection) -> List[Dict]:
    json_list = list()
    for spline_id, time_point, spline in data_axes.all_splines():
        points_x, points_y = spline.get_points_2d()
        json_object = {
            "_time_point_number": time_point.time_point_number(),
            "x_list": points_x,
            "y_list": points_y,
            "z": spline.get_z(),
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


def _encode_connections_to_json(connections: Connections) -> Dict[str, List[List[Position]]]:
    connections_dict = dict()
    for time_point in connections.time_points():
        connections_json = list()
        for position1, position2 in connections.of_time_point(time_point):
            connections_json.append([position1.with_time_point(None), position2.with_time_point(None)])
        connections_dict[str(time_point.time_point_number())] = connections_json
    return connections_dict


def save_data_to_json(experiment: Experiment, json_file_name: str):
    """Saves positions, shapes, scores and links to a JSON file. The file should end with the extension FILE_EXTENSION.
    """
    save_data = {"version": "v1"}

    if experiment.positions.has_positions():
        save_data["positions"] = _encode_positions_and_shapes(experiment.positions, experiment.position_data)

    # Save name
    if experiment.name.has_name():
        save_data["name"] = str(experiment.name)

    # Save links
    if experiment.links.has_links() or experiment.position_data.has_position_data():
        save_data["links"] = _links_to_d3_data(experiment.links, experiment.positions, experiment.position_data)

    # Save scores of families
    scored_families = list(experiment.scores.all_scored_families())
    if len(scored_families) > 0:
        save_data["family_scores"] = scored_families

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
    save_data["image_offsets"] = experiment.images.offsets.to_list()

    _create_parent_directories(json_file_name)
    json_file_name_old = json_file_name + ".OLD"
    if os.path.exists(json_file_name):
        os.rename(json_file_name, json_file_name_old)
    with open(json_file_name, 'w') as handle:
        json.dump(save_data, handle, cls=_MyEncoder)
    if os.path.exists(json_file_name_old):
        os.remove(json_file_name_old)


def _create_parent_directories(file_name: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
