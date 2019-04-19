"""Classes for expressing the positions of positions"""
import json
import os
from json import JSONEncoder
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable

import numpy

from autotrack.core import shape, TimePoint, UserError
from autotrack.core.connections import Connections
from autotrack.core.experiment import Experiment
from autotrack.core.images import ImageOffsets
from autotrack.core.links import Links
from autotrack.core.position_collection import PositionCollection
from autotrack.core.position import Position
from autotrack.core.data_axis import DataAxisCollection, DataAxis
from autotrack.core.resolution import ImageResolution
from autotrack.core.score import ScoredFamily, Score, Family

FILE_EXTENSION = "aut"


def load_positions_and_shapes_from_json(experiment: Experiment, json_file_name: str,
                                        min_time_point: int = 0, max_time_point: int = 5000):
    """Loads a JSON file that contains position positions, with or without shape information."""
    with open(json_file_name) as handle:
        time_points = json.load(handle)
        _parse_shape_format(experiment, time_points, min_time_point, max_time_point)


def _load_guizela_data_file(file_name: str, min_time_point: int, max_time_point: int) -> Experiment:
    """Starting from a random *.p file in a directory, this loads all data according to Guizela's format from that
    directory."""
    experiment = Experiment()
    from autotrack.manual_tracking import guizela_data_importer
    print("File name", file_name)
    guizela_data_importer.add_data_to_experiment(experiment, os.path.dirname(file_name), min_time_point, max_time_point)
    return experiment


def load_data_file(file_name: str, min_time_point: int = 0, max_time_point: int = 5000) -> Experiment:
    """Loads some kind of data file. This should support all data formats of our research group. Raises ValueError if
    the file fails to load."""
    if file_name.lower().endswith("." + FILE_EXTENSION) or file_name.lower().endswith(".json"):
        return _load_json_data_file(file_name, min_time_point, max_time_point)
    elif file_name.lower().endswith(".p"):
        return _load_guizela_data_file(file_name, min_time_point, max_time_point)
    else:
        raise ValueError(f"Cannot load data from file \"{file_name}\": it is of an unknown format")


def _load_json_data_file(file_name: str, min_time_point: int, max_time_point: int):
    """Loads any kind of JSON file."""
    experiment = Experiment()
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
            _parse_data_axes_format(experiment, data["data_axes"], min_time_point, max_time_point)

        if "connections" in data:
            _parse_connections_format(experiment, data["connections"], min_time_point, max_time_point)

        if "family_scores" in data:
            experiment.scores.add_scored_families(data["family_scores"])

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
    return experiment


def load_linking_result(experiment: Experiment, json_file_name: str):
    """Loads a JSON file that is a linking result. Raises ValueError if the file contains no links."""
    new_experiment = load_data_file(json_file_name)
    if not new_experiment.links.has_links():
        raise ValueError("No links found in file", f"The file \"{json_file_name}\" contains no linking data.")
    experiment.links.add_links(new_experiment.links)


def _parse_shape_format(experiment: Experiment, json_structure: Dict[str, List], min_time_point: int, max_time_point: int):
    for time_point_number, raw_positions in json_structure.items():
        time_point_number = int(time_point_number)  # str -> int
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue

        for raw_position in raw_positions:
            position = Position(*raw_position[0:3], time_point_number=time_point_number)
            position_shape = shape.from_list(raw_position[3:])
            experiment.positions.add(position, position_shape)


def _parse_links_format(experiment: Experiment, link_data: Dict[str, Any], min_time_point: int, max_time_point: int):
    """Parses a node_link_graph and adds all links and positions to the experiment."""
    links = Links()
    _add_d3_data(links, link_data, min_time_point, max_time_point)
    positions = experiment.positions
    for position in links.find_all_positions():
        positions.add(position)
    experiment.links.add_links(links)


def _parse_data_axes_format(experiment: Experiment, axes_data: List[Dict], min_time_point: int, max_time_point: int):
    for path_json in axes_data:
        time_point_number = path_json["_time_point_number"]
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue
        path = DataAxis()
        points_x = path_json["x_list"]
        points_y = path_json["y_list"]
        z = path_json["z"]
        for i in range(len(points_x)):
            path.add_point(points_x[i], points_y[i], z)
        path.set_offset(path_json["offset"])
        experiment.data_axes.add_data_axis(TimePoint(time_point_number), path)


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


def _add_d3_data(links: Links, data: Dict, min_time_point: int = -100000, max_time_point: int = 100000):
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
            links.set_position_data(position, data_key, data_value)

    # Add links
    for link in data["links"]:
        source: Position = link["source"]
        target: Position = link["target"]
        if source.time_point_number() < min_time_point or target.time_point_number() < min_time_point \
            or source.time_point_number() > max_time_point or target.time_point_number() > max_time_point:
            continue  # Ignore time points out of range
        links.add_link(source, target)


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


def _links_to_d3_data(links: Links, positions: Iterable[Position]) -> Dict:
    """Return data in D3.js node-link format that is suitable for JSON serialization
    and use in Javascript documents."""
    nodes = list()

    # Save nodes and store extra data
    for position in positions:
        node = {
            "id": position
        }
        for data_name, data_value in links.find_all_data_of_position(position):
            node[data_name] = data_value
        nodes.append(node)

    # Save edges
    edges = list()
    for source, target in links.find_all_links():
        edge = {
            "source": source,
            "target": target
        }
        edges.append(edge)

    return {
        "directed": False,
        "multigraph": False,
        "graph": dict(),
        "nodes": nodes,
        "links": edges
    }


def save_positions_and_shapes_to_json(experiment: Experiment, json_file_name: str):
    """Saves a list of positions to disk."""
    data_structure = _encode_positions_and_shapes(experiment.positions)

    _create_parent_directories(json_file_name)

    json_file_name_old = json_file_name + ".OLD"
    if os.path.exists(json_file_name):
        os.rename(json_file_name, json_file_name_old)
    with open(json_file_name, 'w') as handle:
        json.dump(data_structure, handle, cls=_MyEncoder)
    if os.path.exists(json_file_name_old):
        os.remove(json_file_name_old)


def _encode_positions_and_shapes(positions_and_shapes: PositionCollection):
    data_structure = {}
    for time_point_number in range(positions_and_shapes.first_time_point_number(), positions_and_shapes.last_time_point_number() + 1):
        time_point = TimePoint(time_point_number)
        positions = []
        for position, shape in positions_and_shapes.of_time_point_with_shapes(time_point).items():
            positions.append([position.x, position.y, position.z] + shape.to_list())

        data_structure[str(time_point_number)] = positions
    return data_structure


def save_dataframe_to_csv(data_frame, csv_file_name: str):
    """Saves the data frame to a CSV file, creating necessary parent directories first."""
    from pandas import DataFrame  # Pandas is quite slow to load, so only load it inside this method
    assert isinstance(data_frame, DataFrame)

    _create_parent_directories(csv_file_name)
    try:
        data_frame.to_csv(csv_file_name, index=False)
    except PermissionError as e:
        data_frame.to_csv(csv_file_name + ".ALT", index=False)
        raise e


def _encode_data_axes_to_json(data_axes: DataAxisCollection) -> List[Dict]:
    json_list = list()
    for data_axis, time_point in data_axes.all_data_axes():
        points_x, points_y = data_axis.get_points_2d()
        json_object = {
            "_time_point_number": time_point.time_point_number(),
            "x_list": points_x,
            "y_list": points_y,
            "z": data_axis.get_z(),
            "offset": data_axis.get_offset()
        }
        json_list.append(json_object)
    return json_list


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
        save_data["positions"] = _encode_positions_and_shapes(experiment.positions)

    # Save name
    if experiment.name.has_name():
        save_data["name"] = str(experiment.name)

    # Save links
    if experiment.links.has_links():
        save_data["links"] = _links_to_d3_data(experiment.links, experiment.positions)

    # Save scores of families
    scored_families = list(experiment.scores.all_scored_families())
    if len(scored_families) > 0:
        save_data["family_scores"] = scored_families

    # Save data axes
    if experiment.data_axes.has_axes():
        save_data["data_axes"] = _encode_data_axes_to_json(experiment.data_axes)

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


def load_links_from_json(json_file_name: str, min_time_point: int = 0, max_time_point: int = 5000) -> Links:
    """Loads all links from a file. Links that extend outside the allowed time points are removed."""
    with open(json_file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)
        if data is None:
            raise ValueError
        if "directed" not in data:
            data = data["links"]

        data["nodes"] = [entry for entry in data["nodes"]
                         if min_time_point <= entry["id"].time_point_number() <= max_time_point]
        data["links"] = [entry for entry in data["links"]
                         if min_time_point <= entry["source"].time_point_number() <= max_time_point
                         and min_time_point <= entry["target"].time_point_number() <= max_time_point]

        links = Links()
        _add_d3_data(links, data)
        return links


def _create_parent_directories(file_name: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
