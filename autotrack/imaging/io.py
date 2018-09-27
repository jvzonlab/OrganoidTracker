"""Classes for expressing the positions of particles"""
import json
from json import JSONEncoder
from pathlib import Path
from typing import List, Optional

import numpy
from networkx import node_link_data, node_link_graph, Graph
from pandas import DataFrame

from autotrack.core.experiment import Experiment
from autotrack.core import shape
from autotrack.core.particles import Particle
from autotrack.core.score import ScoredFamily, Score, Family


def load_positions_and_shapes_from_json(experiment: Experiment, json_file_name: str,
                                        min_time_point: Optional[int] = 0, max_time_point: Optional[int] = 5000):
    """Loads all particle positions from a JSON file"""
    experiment.remove_all_particles()
    with open(json_file_name) as handle:
        time_points = json.load(handle)
        for time_point_number, raw_particles in time_points.items():
            time_point_number = int(time_point_number)  # str -> int
            if time_point_number < min_time_point or time_point_number > max_time_point:
                continue

            for raw_particle in raw_particles:
                particle = Particle(*raw_particle[0:3]).with_time_point_number(time_point_number)
                particle_shape = shape.from_list(raw_particle[3:])
                experiment.particles.add(particle, particle_shape)


def load_links_and_scores_from_json(experiment: Experiment, json_file_name: str, links_are_scratch=False):
    with open(json_file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)
        if data is None:
            raise ValueError

        # Read families
        family_scores_list: List[ScoredFamily] = data.get("family_scores", [])
        for scored_family in family_scores_list:
            family = scored_family.family
            experiment.scores.set_family_score(family, scored_family.score)

        # Read graph
        link_data = data if "directed" in data else data["links"]
        graph = node_link_graph(link_data)
        for particle in graph.nodes():
            experiment.add_particle(particle)
        experiment.particle_links_scratch(graph) if links_are_scratch else experiment.particle_links(graph)


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
    data_structure = {}
    for time_point_number in range(experiment.first_time_point_number(), experiment.last_time_point_number() + 1):
        time_point = experiment.get_time_point(time_point_number)
        particles = []
        for particle, shape in experiment.particles.of_time_point_with_shapes(time_point).items():
            particles.append([particle.x, particle.y, particle.z] + shape.to_list())

        data_structure[str(time_point_number)] = particles

    _create_parent_directories(json_file_name)
    with open(json_file_name, 'w') as handle:
        json.dump(data_structure, handle, cls=_MyEncoder)


def save_links_and_scores_to_json(experiment: Experiment, links: Graph, json_file_name: str):
    """1. Saves particle linking data to a JSON file. File follows the d3.js format, like the example here:
        http://bl.ocks.org/mbostock/4062045
    2. Saves mother scores to the same JSON file.
    """
    links_for_json = node_link_data(links)

    families = []
    for time_point in experiment.time_points():
        for scored_family in experiment.scores.of_time_point(time_point):
            families.append(scored_family)

    final_data = {"links": links_for_json, "family_scores": families}
    _create_parent_directories(json_file_name)
    with open(json_file_name, 'w') as handle:
        json.dump(final_data, handle, cls=_MyEncoder)


def save_dataframe_to_csv(data_frame: DataFrame, csv_file_name: str):
    """Saves the data frame to a CSV file, creating necessary parent directories first."""
    _create_parent_directories(csv_file_name)
    try:
        data_frame.to_csv(csv_file_name, index=False)
    except PermissionError as e:
        data_frame.to_csv(csv_file_name + ".ALT", index=False)
        raise e


def load_links_from_json(json_file_name: str) -> Graph:
    with open(json_file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)
        if data is None:
            raise ValueError
        if "directed" not in data:
            data = data["links"]
        graph = node_link_graph(data)
        return graph


def _create_parent_directories(file_name: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)