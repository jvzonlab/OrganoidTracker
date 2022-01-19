# File originally written by Laetitia Hebert in 2018 (Okinawa Institute of Technology, Japan). Modified by Rutger.
#
# The MIT License
#
# Copyright (c) 2018 Okinawa Institute of Science & Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Optional, Iterable, Tuple

import numpy
from numpy import ndarray
import random

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Images
from organoid_tracker.core.position import Position
from organoid_tracker.linking import cell_division_finder, nearest_neighbor_linker
from organoid_tracker.position_detection_cnn.training_data_creator import _ImageWithPositions


class _ImageWithLinks(_ImageWithPositions):
    """Extension of _ImageWithPositions that also includes the linked (target) position."""
    target_xyz_positions = ndarray
    distances = ndarray
    _linked = List[bool]

    def __init__(self, experiment_name: str, images: Images, time_point: TimePoint, xyz_positions: ndarray,
                 target_xyz_positions: ndarray, distances: ndarray, linked: List[bool]):
        # xyz positions: 2D numpy integer array of cell nucleus positions: [ [x,y,z], [x,y,z], ...]
        super().__init__(experiment_name, images, time_point, xyz_positions)
        self._linked = linked
        self.target_xyz_positions = target_xyz_positions
        self.distances = distances


def create_image_with_links_list(experiments: Iterable[Experiment], division_multiplier=10):
    image_with_links_list = list()
    for experiment in experiments:
        # read a complete experiment
        div_positions = cell_division_finder.find_mothers(experiment.links)
        links = experiment.links
        possible_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)

        div_positions_mothers = div_positions.copy()
        for div_pos in div_positions_mothers:
            children = links.find_futures(div_pos)
            div_positions = div_positions.union(children)

        for time_point in experiment.positions.time_points():
            # if there is no next timepoint available end the routine
            if experiment._images.get_image_stack(TimePoint(time_point.time_point_number()+1)) is None:
                break

            # read a single time point and the next one
            positions = experiment.positions.of_time_point(time_point)

            offset = experiment.images.offsets.of_time_point(time_point)
            future_offset = experiment.images.offsets.of_time_point(TimePoint(time_point.time_point_number()+1))

            image_shape = experiment._images.get_image_stack(time_point).shape
            future_image_shape = experiment._images.get_image_stack(TimePoint(time_point.time_point_number()+1)).shape

            positions_xyz = list()
            target_positions_xyz = list()
            distances = list()
            linked = list()

            for position in positions:
                future_possibilities = possible_links.find_futures(position)
                future_link = links.find_futures(position)

                # check if the position is inside the image
                if inside_image(position, offset, image_shape):

                    # is the cell dividing
                    if position in div_positions:
                        repeats = division_multiplier
                    else:
                        repeats=1

                    for i in range(repeats):
                        for future_possibility in future_possibilities:
                            if inside_image(future_possibility, future_offset, future_image_shape):

                                # add positions to list
                                target_positions_xyz.append(
                                    [future_possibility.x - future_offset.x, future_possibility.y - future_offset.y, future_possibility.z - future_offset.z])
                                positions_xyz.append(
                                    [position.x - offset.x, position.y - offset.y, position.z - offset.z])

                                # get distance vector
                                distances.append([(future_possibility.x - position.x), (future_possibility.y - position.y), (future_possibility.z - position.z)])

                                # are the positions linked
                                if future_possibility in future_link:
                                    linked.append(True)
                                else:
                                    linked.append(False)

            # read positions to numpy array
            if len(positions_xyz) > 0:
                positions_xyz = numpy.array(positions_xyz, dtype=numpy.int32)
                target_positions_xyz = numpy.array(target_positions_xyz, dtype=numpy.int32)
                distances = numpy.array(distances, dtype=numpy.int32)
                image_with_links_list.append(
                    _ImageWithLinks(str(experiment.name), experiment.images, time_point, positions_xyz, target_positions_xyz, distances, linked))

    return image_with_links_list


def create_image_with_possible_links_list(experiment: Experiment):
    image_with_links_list = list()
    predicted_links_list = list()

    possible_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)

    for time_point in experiment.positions.time_points():
        if experiment._images.get_image_stack(TimePoint(time_point.time_point_number() + 1)) is None:
            break
        # read a single time point
        positions = experiment.positions.of_time_point(time_point)
        offset = experiment.images.offsets.of_time_point(time_point)
        future_offset = experiment.images.offsets.of_time_point(TimePoint(time_point.time_point_number() + 1))
        image_shape = experiment._images.get_image_stack(time_point).shape
        future_image_shape = experiment._images.get_image_stack(TimePoint(time_point.time_point_number() + 1)).shape

        positions_xyz = list()
        target_positions_xyz = list()
        distances = list()
        linked = list()
        predicted_links = list()

        for position in positions:
            future_possibilities = possible_links.find_futures(position)

            if inside_image(position, offset, image_shape):

                for future_possibility in future_possibilities:
                    if inside_image(future_possibility, future_offset, future_image_shape):

                        target_positions_xyz.append(
                            [future_possibility.x - future_offset.x, future_possibility.y - future_offset.y,
                                future_possibility.z - future_offset.z])
                        positions_xyz.append(
                            [position.x - offset.x, position.y - offset.y, position.z - offset.z])

                        distances.append(
                            [abs(future_possibility.x - position.x), abs(future_possibility.y - position.y),
                             abs(future_possibility.z - position.z)])

                        linked.append(False)

                        predicted_links.append((position, future_possibility))

        # read positions to numpy array
        if len(positions_xyz) > 0:
            positions_xyz = numpy.array(positions_xyz, dtype=numpy.int32)
            target_positions_xyz = numpy.array(target_positions_xyz, dtype=numpy.int32)
            distances = numpy.array(distances, dtype=numpy.int32)

            image_with_links_list.append(
                _ImageWithLinks(str(experiment.name), experiment.images, time_point, positions_xyz,
                target_positions_xyz, distances, linked))

            predicted_links_list.append(predicted_links)

    return image_with_links_list, predicted_links_list, possible_links


def inside_image(position: Position, offset: Images.offsets, image_shape: Tuple[int]):
    inside = False
    if position.x - offset.x < image_shape[2] and position.y - offset.y < image_shape[1]\
            and position.z - offset.z < image_shape[0]:
        inside = True

    if position.x - offset.x >= 0 and position.y - offset.y >= 0 and position.z - offset.z >= 0:
        inside = inside
    else:
        inside = False

    return inside
