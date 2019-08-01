"""Used to create links between positions at different time points. Supports cell divisions. Based on

C. Haubold, J. Ales, S. Wolf, F. A. Hamprecht. A Generalized Successive Shortest Paths Solver for Tracking Dividing
Targets. ECCV 2016 Proceedings.

"""

import dpct
import math
from typing import Dict, List, Iterable

from ai_track.core.links import Links
from ai_track.core.position_collection import PositionCollection
from ai_track.core.position import Position
from ai_track.core.resolution import ImageResolution
from ai_track.core.score import ScoreCollection, Score, ScoredFamily


class _PositionToId:
    __position_to_id: Dict[Position, int]
    __id_to_position: List[Position]

    def __init__(self):
        self.__position_to_id = dict()
        self.__id_to_position = [None, None]  # So the first position will get index 2

    def id(self, position: Position) -> int:
        """Gets the id of the position, or creates a new id of the position doesn't have one yet"""
        id = self.__position_to_id.get(position)
        if id is None:
            id = len(self.__id_to_position)  # This will be the new id
            self.__id_to_position.append(position)
            self.__position_to_id[position] = id
        return id

    def position(self, id: int) -> Position:
        """Gets the position with the given id. Throws IndexError for invalid ids."""
        return self.__id_to_position[id]


def _to_links(position_ids: _PositionToId, results: Dict) -> Links:
    links = Links()

    for entry in results["linkingResults"]:
        if not entry["value"]:
            continue  # Link was not detected
        position1 = position_ids.position(entry["src"])
        position2 = position_ids.position(entry["dest"])
        links.add_link(position1, position2)

    return links


def run(positions: PositionCollection, starting_links: Links, scores: ScoreCollection, resolution: ImageResolution,
        *, link_weight: int, detection_weight: int, division_weight: int, appearance_weight: int,
        dissappearance_weight: int) -> Links:
    """
    Calculates the optimal links, based on the given starting points and weights.
    :param positions: The positions.
    :param starting_links: Basic linking network that includes all possible links.
    :param scores: Scores, for deciding whether something is a cell division.
    :param resolution: Resolution.
    :param link_weight: multiplier for linking features - the higher, the more expensive to create a link.
    :param detection_weight: multiplier for detection of a cell - the higher, the more expensive to omit a cell
    :param division_weight: multiplier for division features - the higher, the cheaper it is to create a cell division
    :param appearance_weight: multiplier for appearance features - the higher, the more expensive it is to create a cell out of nothing
    :param dissappearance_weight: multiplier for disappearance - the higher, the more expensive an end-of-lineage is
    :return:
    """
    position_ids = _PositionToId()
    weights = {"weights": [link_weight, detection_weight, division_weight, appearance_weight, dissappearance_weight]}
    input = _create_dpct_graph(position_ids, starting_links, scores, positions, resolution,
                               positions.first_time_point_number(), positions.last_time_point_number())
    results = dpct.trackFlowBased(input, weights)
    return _to_links(position_ids, results)


def _scores_involving(daughter: Position, scores: Iterable[ScoredFamily]) -> Iterable[ScoredFamily]:
    """Gets all scores where the given position plays the role as a daughter in the given score."""
    for score in scores:
        if daughter in score.family.daughters:
            yield score


def _create_dpct_graph(position_ids: _PositionToId, starting_links: Links, scores: ScoreCollection,
                       shapes: PositionCollection, resolution: ImageResolution,
                       min_time_point: int, max_time_point: int) -> Dict:
    segmentation_hypotheses = []
    for position in starting_links.find_all_positions():
        appearance_penalty = 1 if position.time_point_number() > min_time_point else 0
        disappearance_penalty = 1 if position.time_point_number() < max_time_point else 0

        map = {
            "id": position_ids.id(position),
            "features": [[1.0], [0.0]],  # Assigning a detection to zero cells costs 1, using it is free
            "appearanceFeatures": [[0], [appearance_penalty]],  # Using an appearance is expensive
            "disappearanceFeatures": [[0], [disappearance_penalty]],  # Using a dissappearance is expensive
            "timestep": [position.time_point_number(), position.time_point_number()]
        }

        # Add division score
        division_score = _max_score(scores.of_mother(position))
        if not division_score.is_unlikely_mother():
            map["divisionFeatures"] = [[0], [-division_score.total()]]
        segmentation_hypotheses.append(map)

    linking_hypotheses = []
    for position1, position2 in starting_links.find_all_links():
        # Make sure position1 is earlier in time
        if position1.time_point_number() > position2.time_point_number():
            position1, position2 = position2, position1

        volume1, volume2 = shapes.get_shape(position1).volume(), shapes.get_shape(position2).volume()
        link_penalty = position1.distance_um(position2, resolution)
        link_penalty += (abs(volume1 - volume2) ** (1 / 3)) * resolution.pixel_size_x_um

        mother_score = _max_score(_scores_involving(position2, scores.of_mother(position1)))

        if not mother_score.is_unlikely_mother():
            link_penalty /= 2
        linking_hypotheses.append({
            "src": position_ids.id(position1),
            "dest": position_ids.id(position2),
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
