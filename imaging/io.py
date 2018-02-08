"""Classes for expressing the positions of particles"""
import json
from json import JSONEncoder
from imaging import Experiment, Particle
from networkx import node_link_data, node_link_graph, Graph


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

        return JSONEncoder.default(self, o)


def _my_decoder(json_object):
    if 'x' in json_object and 'y' in json_object and 'z' in json_object:
        particle = Particle(json_object['x'], json_object['y'], json_object['z'])
        if '_frame_number' in json_object:
            particle.frame_number(json_object['_frame_number'])
        return particle
    return json_object

def save_links_to_json(links: Graph, json_file_name: str):
    """Saves particle linking data to a JSON file. File follows the d3.js format, like the example here:
    http://bl.ocks.org/mbostock/4062045 """
    data = node_link_data(links)
    with open(json_file_name, 'w') as handle:
        json.dump(data, handle, cls=_MyEncoder)

def load_links_from_json(experiment: Experiment, json_file_name: str):
    with open(json_file_name) as handle:
        data = json.load(handle, object_hook=_my_decoder)
        if data == None:
            raise ValueError
        graph = node_link_graph(data)
        experiment.particle_links(graph)