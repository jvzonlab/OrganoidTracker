"""Classes for expressing the positions of particles"""
import json
import os
from json import JSONEncoder
from pathlib import Path
from typing import List, Dict, Any

import numpy
from networkx import node_link_data, node_link_graph, Graph
from pandas import DataFrame

from autotrack.core import shape, TimePoint
from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.resolution import ImageResolution
from autotrack.core.score import ScoredFamily, Score, Family

FILE_EXTENSION = "aut"


def load_positions_and_shapes_from_json(experiment: Experiment, json_file_name: str,
                                        min_time_point: int = 0, max_time_point: int = 5000):
    """Loads a JSON file that contains particle positions, with or without shape information."""
    with open(json_file_name) as handle:
        time_points = json.load(handle)
        _parse_shape_format(experiment, time_points, min_time_point, max_time_point)


def _load_guizela_data_file(file_name: str, min_time_point: int, max_time_point: int) -> Experiment:
    """Starting from a random *.p file in a directory, this loads all data according to Guizela's format from that
    directory."""
    experiment = Experiment()
    from autotrack.manual_tracking import guizela_data_importer
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
        raise ValueError("Cannot load data from file " + file_name)


def _load_json_data_file(file_name: str, min_time_point: int, max_time_point: int):
    """Loads any kind of JSON file."""
    experiment = Experiment()
    with open(file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)

        if "version" not in data and "links" not in data:
            # We don't have a general data file, but a specialized one
            if "directed" in data:
                # File is a linking result file
                _parse_links_format(experiment, data)
            else:  # file is a position/shape file
                _parse_shape_format(experiment, data, min_time_point, max_time_point)
            return experiment

        if data.get("version", "v1") != "v1":
            raise ValueError("Unknown data version", "This program is not able to load data of version "
                             + str(data["version"]) + ".")

        if "shapes" in data:
            _parse_shape_format(experiment, data["shapes"], min_time_point, max_time_point)

        if "family_scores" in data:
            experiment.scores.add_scored_families(data["family_scores"])

        if "links" in data:
            _parse_links_format(experiment, data["links"])
        elif "links_scratch" in data:  # Deprecated, was used back when experiments could hold multiple linking sets
            _parse_links_format(experiment, data["links_scratch"])
        elif "links_baseline" in data:  # Deprecated, was used back when experiments could hold multiple linking sets
            _parse_links_format(experiment, data["links_baseline"])

        if "image_resolution" in data:
            x_res = data["image_resolution"]["x_um"]
            y_res = data["image_resolution"]["y_um"]
            z_res = data["image_resolution"]["z_um"]
            t_res = data["image_resolution"]["t_m"]
            experiment.image_resolution(ImageResolution(x_res, y_res, z_res, t_res))
    return experiment


def load_linking_result(experiment: Experiment, json_file_name: str):
    """Loads a JSON file that is a linking result."""
    with open(json_file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)
        if data is None:
            raise ValueError
        _parse_links_format(experiment, data)


def _parse_shape_format(experiment: Experiment, json_structure: Dict[str, List], min_time_point: int, max_time_point: int):
    for time_point_number, raw_particles in json_structure.items():
        time_point_number = int(time_point_number)  # str -> int
        if time_point_number < min_time_point or time_point_number > max_time_point:
            continue

        for raw_particle in raw_particles:
            particle = Particle(*raw_particle[0:3]).with_time_point_number(time_point_number)
            particle_shape = shape.from_list(raw_particle[3:])
            experiment.particles.add(particle, particle_shape)


def _parse_links_format(experiment: Experiment, link_data: Dict[str, Any]):
    """Parses a node_link_graph and adds all links and particles to the experiment."""
    graph = node_link_graph(link_data)
    for particle in graph.nodes():
        experiment.add_particle(particle)
    experiment.links.add_links(graph)


class _MyEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Particle) or isinstance(o, Score):
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
        particle = Particle(json_object['x'], json_object['y'], json_object['z'])
        if '_time_point_number' in json_object:
            particle.with_time_point_number(json_object['_time_point_number'])
        return particle
    if 'scores' in json_object and 'mother' in json_object and 'daughter1' in json_object:
        mother = json_object["mother"]
        daughter1 = json_object["daughter1"]
        daughter2 = json_object["daughter2"]
        family = Family(mother, daughter1, daughter2)
        score = Score(**json_object["scores"])
        return ScoredFamily(family, score)
    return json_object


def save_links_to_json(links: Graph, json_file_name: str):
    """Saves particle linking data to a JSON file. File follows the d3.js format, like the example here:
    http://bl.ocks.org/mbostock/4062045 """
    data = node_link_data(links)

    _create_parent_directories(json_file_name)
    with open(json_file_name, 'w') as handle:
        json.dump(data, handle, cls=_MyEncoder)


def save_positions_and_shapes_to_json(experiment: Experiment, json_file_name: str):
    """Saves a list of particles to disk."""
    data_structure = _encode_positions_and_shapes(experiment.particles)

    _create_parent_directories(json_file_name)
    with open(json_file_name, 'w') as handle:
        json.dump(data_structure, handle, cls=_MyEncoder)


def _encode_positions_and_shapes(particles_and_shapes: ParticleCollection):
    data_structure = {}
    for time_point_number in range(particles_and_shapes.first_time_point_number(), particles_and_shapes.last_time_point_number() + 1):
        time_point = TimePoint(time_point_number)
        particles = []
        for particle, shape in particles_and_shapes.of_time_point_with_shapes(time_point).items():
            particles.append([particle.x, particle.y, particle.z] + shape.to_list())

        data_structure[str(time_point_number)] = particles
    return data_structure


def save_dataframe_to_csv(data_frame: DataFrame, csv_file_name: str):
    """Saves the data frame to a CSV file, creating necessary parent directories first."""
    _create_parent_directories(csv_file_name)
    try:
        data_frame.to_csv(csv_file_name, index=False)
    except PermissionError as e:
        data_frame.to_csv(csv_file_name + ".ALT", index=False)
        raise e


def save_data_to_json(experiment: Experiment, json_file_name: str):
    """Saves positions, shapes, scores and links to a JSON file. The file should end with the extension FILE_EXTENSION.
    """
    save_data = {
        "version": "v1",
        "shapes": _encode_positions_and_shapes(experiment.particles)}

    # Save links
    if experiment.links.graph is not None:
        save_data["links"] = node_link_data(experiment.links.graph)

    # Save scores of families
    scored_families = list(experiment.scores.all_scored_families())
    if len(scored_families) > 0:
        save_data["family_scores"] = scored_families

    # Save image resolution
    try:
        resolution = experiment.image_resolution()
        save_data["image_resolution"] = {"x_um": resolution.pixel_size_zyx_um[2],
                                         "y_um": resolution.pixel_size_zyx_um[1],
                                         "z_um": resolution.pixel_size_zyx_um[0],
                                         "t_m": resolution.pixel_size_zyx_um[0]}
    except ValueError:
        pass

    _create_parent_directories(json_file_name)
    with open(json_file_name, 'w') as handle:
        json.dump(save_data, handle, cls=_MyEncoder)


def load_links_from_json(json_file_name: str, min_time_point: int = 0, max_time_point: int = 5000) -> Graph:
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

        return node_link_graph(data)


def _create_parent_directories(file_name: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
