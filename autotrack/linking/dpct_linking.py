import dpct
import math
from pprint import pprint
from typing import Dict, List, Iterable, Optional

from networkx import Graph

from autotrack.core.experiment import Experiment
from autotrack.core.particles import Particle, ParticleCollection
from autotrack.core.score import ScoreCollection, Score, ScoredFamily


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


def run(particles: ParticleCollection, starting_links: Graph, scores: ScoreCollection):
    particle_ids = _ParticleToId()
    weights = {"weights": [
        10,  # multiplier for linking features?
        150,  # multiplier for detection of a cell - the higher, the more expensive to omit a cell
        30,  # multiplier for division features - the higher, the cheaper it is to create a cell division
        150,  # multiplier for appearance features - the higher, the more expensive it is to create a cell out of nothing
        100]}  # multiplier for disappearance - the higher, the more expensive an end-of-lineage is
    input = _create_dpct_graph(particle_ids, starting_links, scores, particles,
                               particles.first_time_point_number(), particles.last_time_point_number())
    results = dpct.trackFlowBased(input, weights)
    return _to_graph(particle_ids, results)


def _scores_involving(daughter: Particle, scores: Iterable[ScoredFamily]) -> Iterable[ScoredFamily]:
    """Gets all scores where the given particle plays the role as a daughter in the given score."""
    for score in scores:
        if daughter in score.family.daughters:
            yield score


def _create_dpct_graph(particle_ids: _ParticleToId, starting_links: Graph, scores: ScoreCollection,
                       shapes: ParticleCollection, min_time_point: int, max_time_point: int) -> Dict:
    segmentation_hypotheses = []
    particle: Particle
    for particle in starting_links.nodes:
        appearance_penalty = 1 if particle.time_point_number() > min_time_point else 0
        disappearance_penalty = 1 if particle.time_point_number() < max_time_point else 0

        map = {
            "id": particle_ids.id(particle),
            "features": [[1.0], [0.0]],  # Assigning a detection to zero cells costs 1, using it is free
            "appearanceFeatures": [[0], [appearance_penalty]],  # Using an appearance is expensive
            "disappearanceFeatures": [[0], [disappearance_penalty]],  # Using a dissappearance is expensive
            "timestep": [particle.time_point_number(), particle.time_point_number()]
        }

        # Add division score
        division_score = _max_score(scores.of_mother(particle))
        if not division_score.is_unlikely_mother():
            map["divisionFeatures"] = [[0], [-division_score.total()]]
        segmentation_hypotheses.append(map)

    linking_hypotheses = []
    particle1: Particle
    particle2: Particle
    for particle1, particle2 in starting_links.edges:
        # Make sure particle1 is earlier in time
        if particle1.time_point_number() > particle2.time_point_number():
            particle1, particle2 = particle2, particle1

        volume1, volume2 = shapes.get_shape(particle1).volume(), shapes.get_shape(particle2).volume()
        link_penalty = math.sqrt(particle1.distance_squared(particle2))
        link_penalty += abs(volume1 - volume2) ** (1 / 3)

        mother_score = _max_score(_scores_involving(particle2, scores.of_mother(particle1)))

        if not mother_score.is_unlikely_mother():
            link_penalty /= 2
        linking_hypotheses.append({
            "src": particle_ids.id(particle1),
            "dest": particle_ids.id(particle2),
            "features": [[0],  # Sending zero cells through the link costs nothing
                         [link_penalty]  # Sending one cell through the link costs this
                         ]
        })

    return {
        "settings": {
            "statesShareWeights": True
        },

        "segmentationHypotheses": segmentation_hypotheses,
        "linkingHypotheses": linking_hypotheses
    }


class _ZeroScore(Score):
    def __setattr__(self, key, value):
        raise RuntimeError("Cannot change the zero score")


_ZERO_SCORE = _ZeroScore()


def _max_score(scored_family: Iterable[ScoredFamily]) -> Score:
    """Returns the highest score from the collection, or zero if the collection is empty."""
    max_score = None
    for family in scored_family:
        if max_score is None or family.score.total() > max_score.total():
            max_score = family.score
    if max_score is None:
        return _ZERO_SCORE
    return max_score
