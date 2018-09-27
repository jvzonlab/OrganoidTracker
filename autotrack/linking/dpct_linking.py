from typing import Dict

from networkx import Graph

from autotrack.core import Particle


class _ParticleToId:
    __dict: Dict[Particle, int]
    __next_id: int = 2

    def __init__(self):
        self.__dict = dict()

    def get(self, particle: Particle) -> int:
        """Gets the id of the particle, or creates a new id of the particle doesn't have one yet"""
        id = self.__dict.get(particle)
        if id is None:
            id = self.__next_id
            self.__next_id += 1
            self.__dict[particle] = id
        return id


def __create_dpct_graph(links_hypothesis: Graph) -> Dict:
    particle_ids = _ParticleToId()
    dictionary = {
        "settings": {
            "statesShareWeights": True
        }
    }

    segmentation_hypotheses = []

    particle: Particle
    for particle in links_hypothesis.nodes:
        segmentation_hypotheses.append({
            "id": particle_ids.get(particle),
            "features": [[1.0], [0.0]],
            "divisionFeatures": [[0.0], [-5.0]],
            "appearanceFeatures": [[0], [0]],
            "disappearanceFeatures": [[0], [50]],
            "timestep": [particle.time_point_number(), particle.time_point_number()]
        })


    return dictionary
