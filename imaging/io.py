"""Classes for expressing the positions of particles"""
import json
from json import JSONEncoder

from pandas import DataFrame

from imaging import Experiment, Particle
from networkx import node_link_data, node_link_graph, Graph
from typing import List
import numpy
from pathlib import Path
import os

def load_positions_from_json(experiment: Experiment, json_file_name: str):
    """Loads all particle positions from a JSON file"""
    with open(json_file_name) as handle:
        time_points = json.load(handle)
        for time_point, raw_particles in time_points.items():
            experiment.add_particles(int(time_point), raw_particles)


class _MyEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Particle):
            return o.__dict__

        if isinstance(o, numpy.int32):
            return numpy.asscalar(o)

        return JSONEncoder.default(self, o)


def _my_decoder(json_object):
    if 'x' in json_object and 'y' in json_object and 'z' in json_object:
        particle = Particle(json_object['x'], json_object['y'], json_object['z'])
        if '_time_point_number' in json_object:
            particle.with_time_point_number(json_object['_time_point_number'])
        return particle
    return json_object


def save_links_to_json(links: Graph, json_file_name: str):
    """Saves particle linking data to a JSON file. File follows the d3.js format, like the example here:
    http://bl.ocks.org/mbostock/4062045 """
    data = node_link_data(links)

    _create_parent_directories(json_file_name)
    with open(json_file_name, 'w') as handle:
        json.dump(data, handle, cls=_MyEncoder)


def save_positions_to_json(experiment: Experiment, json_file_name: str):
    """Saves a list of particles to disk."""
    data_structure = {}
    for time_point_number in range(experiment.first_time_point_number(), experiment.last_time_point_number() + 1):
        time_point = experiment.get_time_point(time_point_number)
        particles = [(p.x, p.y, p.z) for p in time_point.particles()]
        data_structure[str(time_point_number)] = particles

    _create_parent_directories(json_file_name)
    with open(json_file_name, 'w') as handle:
        json.dump(data_structure, handle, cls=_MyEncoder)

def save_dataframe_to_csv(data_frame: DataFrame, csv_file_name: str):
    """Saves the data frame to a CSV file, creating necessary parent directories first."""
    _create_parent_directories(csv_file_name)
    data_frame.to_csv(csv_file_name, index=False)


def load_links_from_json(json_file_name: str) -> Graph:
    with open(json_file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)
        if data is None:
            raise ValueError
        graph = node_link_graph(data)
        return graph


def _create_parent_directories(file_name: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)