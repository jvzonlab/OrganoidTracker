"""Classes for expressing the positions of particles"""
import json
from json import JSONEncoder
from imaging import Experiment, Particle
from networkx import node_link_data, node_link_graph, Graph
from typing import List
import numpy


def load_positions_from_json(experiment: Experiment, json_file_name: str):
    """Loads all particle positions from a JSON file"""
    with open(json_file_name) as handle:
        frames = json.load(handle)
        for frame, raw_particles in frames.items():
            experiment.add_particles(int(frame), raw_particles)


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
        if '_frame_number' in json_object:
            particle.with_frame_number(json_object['_frame_number'])
        return particle
    return json_object


def save_links_to_json(links: Graph, json_file_name: str):
    """Saves particle linking data to a JSON file. File follows the d3.js format, like the example here:
    http://bl.ocks.org/mbostock/4062045 """
    data = node_link_data(links)
    with open(json_file_name, 'w') as handle:
        json.dump(data, handle, cls=_MyEncoder)


def save_positions_to_json(experiment: Experiment, json_file_name: str):
    """Saves a list of particles to disk."""
    data_structure = {}
    for frame_number in range(experiment.first_frame_number(), experiment.last_frame_number() + 1):
        frame = experiment.get_frame(frame_number)
        particles = [(p.x, p.y, p.z) for p in frame.particles()]
        data_structure[str(frame_number)] = particles
    with open(json_file_name, 'w') as handle:
        json.dump(data_structure, handle, cls=_MyEncoder)


def load_links_from_json(json_file_name: str) -> Graph:
    with open(json_file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)
        if data == None:
            raise ValueError
        graph = node_link_graph(data)
        return graph