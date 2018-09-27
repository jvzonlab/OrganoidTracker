from typing import Dict

from networkx import Graph

from autotrack.core.particles import Particle


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


def __create_dpct_graph(starting_links: Graph, min_time_point: int, max_time_point: int) -> Dict:
    particle_ids = _ParticleToId()

    segmentation_hypotheses = []
    particle: Particle
    for particle in starting_links.nodes:
        appearance_penalty = 50 if particle.time_point_number() > min_time_point else 0
        disappearance_penalty = 50 if particle.time_point_number() < max_time_point else 0

        segmentation_hypotheses.append({
            "id": particle_ids.get(particle),
            "features": [[1.0], [0.0]],
            "divisionFeatures": [[0.0], [-5.0]],
            "appearanceFeatures": [[0], [appearance_penalty]],
            "disappearanceFeatures": [[0], [disappearance_penalty]],
            "timestep": [particle.time_point_number(), particle.time_point_number()]
        })

    linking_hypotheses = []
    particle1: Particle
    particle2: Particle
    for particle1, particle2 in starting_links.edges:
        # Make sure particle1 is earlier in time
        if particle1.time_point_number() > particle2.time_point_number():
            particle1, particle2 = particle2, particle1

        distance_squared = particle1.distance_squared(particle2)
        linking_hypotheses.append({
            "src": particle_ids.get(particle1),
            "dest": particle_ids.get(particle2),
            "features": [[0], [-distance_squared]]
        })

    return {
        "settings": {
            "statesShareWeights": True
        },

        "segmentationHypotheses": segmentation_hypotheses,
        "linkingHypotheses": linking_hypotheses
    }

