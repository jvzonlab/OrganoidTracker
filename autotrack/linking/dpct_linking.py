import dpct
from pprint import pprint
from typing import Dict, List

from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle


class _ParticleToId:
    __particle_to_id: Dict[Particle, int]
    __id_to_particle: List[Particle]

    def __init__(self):
        self.__particle_to_id = dict()
        self.__id_to_particle = [None, None]  # So the first particle will get index 2

    def id(self, particle: Particle) -> int:
        """Gets the id of the particle, or creates a new id of the particle doesn't have one yet"""
        id = self.__particle_to_id.get(particle)
        if id is None:
            id = len(self.__id_to_particle)  # This will be the new id
            self.__id_to_particle.append(particle)
            self.__particle_to_id[particle] = id
        return id

    def particle(self, id: int) -> Particle:
        """Gets the particle with the given id. Throws IndexError for invalid ids."""
        return self.__id_to_particle[id]


def _to_graph(particle_ids: _ParticleToId, results: Dict) -> Graph:
    graph = Graph()

    # Add nodes
    for entry in results["detectionResults"]:
        if not entry["value"]:
            continue  # Cell was not detected
        particle = particle_ids.particle(entry["id"])
        graph.add_node(particle)

    # Add edges
    for entry in results["linkingResults"]:
        if not entry["value"]:
            continue  # Link was not detected
        particle1 = particle_ids.particle(entry["src"])
        particle2 = particle_ids.particle(entry["dest"])
        graph.add_edge(particle1, particle2)

    return graph


def run(experiment: Experiment, starting_links: Graph):
    particle_ids = _ParticleToId()
    weights = {"weights": [
        1,  # multiplier for linking features?
        100,  # multiplier for detection of a cell - the higher, the more expensive to omit a cell
        1,  # multiplier for division features - the higher, the cheaper it is to create a cell division
        100,  # multiplier for appearance features - the higher, the more expensive it is to create a cell out of nothing
        100]}  # multiplier for disappearance - the higher, the more expensive an end-of-lineage is
    input = _create_dpct_graph(particle_ids, starting_links,
                               experiment.first_time_point_number(), experiment.last_time_point_number())
    results = dpct.trackFlowBased(input, weights)
    return _to_graph(particle_ids, results)


def _create_dpct_graph(particle_ids: _ParticleToId, starting_links: Graph, min_time_point: int, max_time_point: int) -> Dict:
    segmentation_hypotheses = []
    particle: Particle
    for particle in starting_links.nodes:
        appearance_penalty = 1 if particle.time_point_number() > min_time_point else 0
        disappearance_penalty = 1 if particle.time_point_number() < max_time_point else 0

        segmentation_hypotheses.append({
            "id": particle_ids.id(particle),
            "features": [[1.0], [0.0]],
            "divisionFeatures": [[0.0], [-1.0]],
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
            "src": particle_ids.id(particle1),
            "dest": particle_ids.id(particle2),
            "features": [[0], [distance_squared]]
        })

    return {
        "settings": {
            "statesShareWeights": True
        },

        "segmentationHypotheses": segmentation_hypotheses,
        "linkingHypotheses": linking_hypotheses
    }

