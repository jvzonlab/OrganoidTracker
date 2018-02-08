"""Classes for expressing the positions of particles"""
import json
from imaging import Experiment


def load_positions_from_json(experiment: Experiment, json_file_name: str):
    """Loads all positions from a JSON file"""
    with open(json_file_name) as handle:
        frames = json.load(handle)
        for frame, raw_particles in frames.items():
            experiment.add_particles(int(frame), raw_particles)

