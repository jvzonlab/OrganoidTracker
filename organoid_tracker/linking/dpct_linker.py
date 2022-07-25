"""Used to create links between positions at different time points. Supports cell divisions. Based on

C. Haubold, J. Ales, S. Wolf, F. A. Hamprecht. A Generalized Successive Shortest Paths Solver for Tracking Dividing
Targets. ECCV 2016 Proceedings.

"""

import dpct
import numpy as np
import math
from typing import Dict, List, Iterable, Tuple

from organoid_tracker.core.link_data import LinkData
from organoid_tracker.core.links import Links
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.score import ScoreCollection, Score, ScoredFamily
from organoid_tracker.linking_analysis import linking_markers


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

def run(positions: PositionCollection, position_data: PositionData, starting_links: Links, link_data: LinkData,
            *, link_weight: int, detection_weight: int, division_weight: int, appearance_weight: int,
            dissappearance_weight: int) -> Tuple[Links, Links]:
    """
    Calculates the optimal links, based on the given starting points and weights.
    :param positions: The positions.
    :param position_data: Metadata for the positions. Must contain 'division_penalty', 'appearance_penalty',
    'dissappearance_penalty' for all positions.
    :param starting_links: Basic linking network that includes all possible links.
    :param link_data: Metadata for the links. Must contain 'link_penalty' for all potential links.
    :param link_weight: multiplier for linking features - the higher, the more expensive to create a link.
    :param detection_weight: multiplier for detection of a cell - the higher, the more expensive to omit a cell
    :param division_weight: multiplier for division features - the higher, the cheaper it is to create a cell division
    :param appearance_weight: multiplier for appearance features - the higher, the more expensive it is to create a cell out of nothing
    :param dissappearance_weight: multiplier for disappearance - the higher, the more expensive an end-of-lineage is
    :return:
    """
    position_ids = _PositionToId()
    input, has_possible_divisions, naive_links = _create_dpct_graph(position_ids, starting_links, position_data, link_data,
                                        positions.first_time_point_number(), positions.last_time_point_number())

    if has_possible_divisions:
        weights = {"weights": [link_weight, detection_weight, division_weight, appearance_weight, dissappearance_weight]}
    else:
        weights = {"weights": [link_weight, detection_weight, appearance_weight, dissappearance_weight]}
    results = dpct.trackFlowBased(input, weights)
    return _to_links(position_ids, results), naive_links


def _create_dpct_graph(position_ids: _PositionToId, starting_links: Links,
                       position_data: PositionData, link_data: LinkData,
                       min_time_point: int, max_time_point: int, division_penalty_cut_off = 1, ignore_penalty = 2.0, penalty_difference_cut_off = 4.0) -> Tuple[Dict, bool, Links]:
    """Creates the linking network. Returns the network and whether there are possible divisions."""
    created_possible_division = False

    naive_links = Links()
    segmentation_hypotheses = []

    for position in starting_links.find_all_positions():
        position_data.set_position_data(position, data_name='min_in_link_penalty', value=10)
        position_data.set_position_data(position, data_name='min_out_link_penalty', value=10)

        # find (dis)appearance penalty
        appearance_penalty = position_data.get_position_data(position, data_name='appearance_penalty') if position.time_point_number() > min_time_point else 0
        disappearance_penalty = position_data.get_position_data(position, data_name='disappearance_penalty') if position.time_point_number() < max_time_point else 0

        # find division penalty
        division_penalty = position_data.get_position_data(position, data_name='division_penalty')

        # check if all the scores are there!
        if disappearance_penalty is None:
            print('alarm')
        if division_penalty is None:
            print('missing')

        if division_penalty < division_penalty_cut_off:
            map = {
                "id": position_ids.id(position),
                "features": [[ignore_penalty], [0]],  # Assigning a detection to zero cells costs, using it is free
                "appearanceFeatures": [[0], [appearance_penalty]],  # Using an appearance is expensive
                "disappearanceFeatures": [[0], [disappearance_penalty]],  # Using a dissappearance is expensive
                "divisionFeatures": [[0], [division_penalty]],
                "timestep": [position.time_point_number(), position.time_point_number()]
            }
            created_possible_division = True
        else:
            map = {
                "id": position_ids.id(position),
                "features": [[ignore_penalty], [0]],  # Assigning a detection to zero cells costs, using it is free
                "appearanceFeatures": [[0], [appearance_penalty]],  # Using an appearance is expensive
                "disappearanceFeatures": [[0], [disappearance_penalty]],  # Using a dissappearance is expensive
                "timestep": [position.time_point_number(), position.time_point_number()]
            }
        segmentation_hypotheses.append(map)

    linking_hypotheses = []

    # first cycle over all the links
    for position1, position2 in starting_links.find_all_links():
        # Make sure position1 is earlier in time
        if position1.time_point_number() > position2.time_point_number():
            print('happens?')
            position1, position2 = position2, position1

        # get link penalty
        link_penalty = link_data.get_link_data(position1, position2, data_name="link_penalty")
        # is it there?
        if link_penalty is None:
            print("link data missing")
            link_penalty = 2

        # determine the lowest in and out going link penalty for every cell
        if link_penalty < position_data.get_position_data(position2, 'min_in_link_penalty'):
            position_data.set_position_data(position2, 'min_in_link_penalty', link_penalty)

        if link_penalty < position_data.get_position_data(position1, 'min_out_link_penalty'):
            position_data.set_position_data(position1, 'min_out_link_penalty', link_penalty)

    j= 0
    # second cycle over all links
    for position1, position2 in starting_links.find_all_links():
        # Make sure position1 is earlier in time
        if position1.time_point_number() > position2.time_point_number():
            print('happens?')
            position1, position2 = position2, position1

        link_penalty = link_data.get_link_data(position1, position2, data_name="link_penalty")
        if link_penalty is None:
            print("link data missing" + str(j))
            j = j + 1
            link_penalty = 2

        # if the link penalty is much smaller then other options we can ignore it
        if (link_penalty < position_data.get_position_data(position2, 'min_in_link_penalty') + penalty_difference_cut_off): # and \
                #(link_penalty < position_data.get_position_data(position1, 'min_out_link_penalty') + penalty_difference_cut_off):

            naive_links.add_link(position1, position2)
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
        "linkingHypotheses": linking_hypotheses,
    }, created_possible_division, naive_links


def calculate_appearance_penalty(experiment, min_appearance_probability, name='appearance_penalty', buffer_distance = 5.0):
    # go over all timepoints
    for time_point in experiment.positions.time_points():
        positions = experiment.positions.of_time_point(time_point)
        offset = experiment.images.offsets.of_time_point(time_point)
        image_shape = experiment._images.get_image_stack(time_point).shape
        resolution = experiment.images.resolution()

        # and all positions
        for position in positions:
            # get distances to the image edges in x, y and z
            distances = [(image_shape[2] - position.x + offset.x) * resolution.pixel_size_x_um,
                         (image_shape[1] - position.y + offset.y) * resolution.pixel_size_y_um,
                         (image_shape[0] - position.z + offset.z) * resolution.pixel_size_z_um,
                         (position.x - offset.x) * resolution.pixel_size_x_um,
                         (position.y - offset.y) * resolution.pixel_size_y_um,
                         (position.z - offset.z) * resolution.pixel_size_z_um]
            # get minimum distance
            min_distance = min(distances)

            # calculate (dis)appearance probability
            if min_distance < buffer_distance:
                appearance_probability = 0.5 * (1 - min_distance/buffer_distance) + min_appearance_probability
            else:
                appearance_probability = min_appearance_probability

            appearance_penalty = - np.log10(appearance_probability) + np.log10(1-appearance_probability)

            experiment.position_data.set_position_data(position, data_name=name, value=appearance_penalty)

    return experiment

