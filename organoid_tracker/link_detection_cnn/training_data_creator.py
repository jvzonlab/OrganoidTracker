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


def create_image_with_links_list(experiments: Iterable[Experiment], division_multiplier=5, mid_distance_multiplier=5,
                                 loose_end_multiplier=0, multiple_options_multiplier=0,
                                 div_window=3, medium_distance=(3, 7), min_cases=0):
    image_with_links_list = list()

    print('analyzing links...')
    for experiment in experiments:
        # read a complete experiment
        resolution = experiment.images.resolution()
        div_positions = cell_division_finder.find_mothers(experiment.links, exclude_multipolar=False)
        links = experiment.links
        possible_links = nearest_neighbor_linker.nearest_neighbor(experiment, tolerance=2)

        div_positions_plus_window = set()

        for div_pos in div_positions:
            children = links.find_futures(div_pos)

            for child in children:
                futures = list(links.iterate_to_future(child))
                div_positions_plus_window = div_positions_plus_window.union(set(futures[:div_window]))

            pasts = list(links.iterate_to_past(div_pos))
            div_positions_plus_window = div_positions_plus_window.union(set(pasts[:div_window + 1]))

        # find loose ends, possibly extruding cells
        loose_ends = list(experiment.links.find_disappeared_positions(
            time_point_number_to_ignore=experiment.last_time_point_number()))
        end_positions_plus_window = set()

        if loose_end_multiplier > 0:
            for end_pos in loose_ends:
                pasts = list(links.iterate_to_past(end_pos))
                loose_end_window = 0
                end_positions_plus_window = end_positions_plus_window.union(
                    set(pasts[:min(loose_end_window + 1, len(pasts))]))

        # find positions with multiple out-links
        multiple_options_positions = []

        if multiple_options_multiplier > 0:

            for position in experiment.positions:
                futures = list(possible_links.find_futures(position))

                if len(futures) > 1:
                    multiple_options_positions.append(position)

        for time_point in experiment.positions.time_points():
            # if there is no next timepoint available end the routine
            if experiment._images.get_image_stack(TimePoint(time_point.time_point_number() + 1)) is None:
                break

            # counts if events of interest happen at this timepoint.
            # First three were usefull for diagnostics/trouble shooting.
            mid_counter = 0
            loose_counter = 0
            multiple_counter = 0
            upsampled_case = 0

            # read a single time point and the next one
            positions = experiment.positions.of_time_point(time_point)

            offset = experiment.images.offsets.of_time_point(time_point)
            future_offset = experiment.images.offsets.of_time_point(TimePoint(time_point.time_point_number() + 1))

            image_shape = experiment._images.get_image_stack(time_point).shape
            future_image_shape = experiment._images.get_image_stack(TimePoint(time_point.time_point_number() + 1)).shape

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
                    if position in div_positions_plus_window:
                        repeats = division_multiplier
                    else:
                        repeats = 1

                    # end point
                    if position in end_positions_plus_window:
                        repeats = loose_end_multiplier
                        loose_counter = loose_counter + 1

                    # multiple options
                    if position in multiple_options_positions:
                        repeats = multiple_options_multiplier
                        multiple_counter = multiple_counter + 1

                    for future_possibility in future_possibilities:
                        if inside_image(future_possibility, future_offset, future_image_shape):

                            repeats_link = repeats

                            if future_possibility in div_positions_plus_window:
                                repeats_link = division_multiplier

                            distance_sq = (((future_possibility.x - position.x) * resolution.pixel_size_x_um) ** 2 +
                                           ((future_possibility.y - position.y) * resolution.pixel_size_y_um) ** 2 +
                                           ((future_possibility.z - position.z) * resolution.pixel_size_z_um) ** 2)

                            # is the distance travelled not very informative (not very far away not very close)?
                            if (distance_sq > medium_distance[0] ** 2) and (distance_sq < medium_distance[1] ** 2):
                                repeats_link = mid_distance_multiplier

                                mid_counter = mid_counter + 1

                            if repeats_link > 1:
                                upsampled_case = upsampled_case + 1

                            for i in range(repeats_link):
                                # add positions to list
                                target_positions_xyz.append(
                                    [future_possibility.x - future_offset.x, future_possibility.y - future_offset.y,
                                     future_possibility.z - future_offset.z])
                                positions_xyz.append(
                                    [position.x - offset.x, position.y - offset.y, position.z - offset.z])

                                # get distance vector
                                distances.append(
                                    [(future_possibility.x - position.x), (future_possibility.y - position.y),
                                     (future_possibility.z - position.z)])

                                # are the positions linked
                                if future_possibility in future_link:
                                    linked.append(True)
                                else:
                                    linked.append(False)

            # read positions to numpy array
            if len(positions_xyz) > 0:
                repeat_image = 1

                if upsampled_case >= min_cases:

                    positions_xyz = numpy.array(positions_xyz, dtype=numpy.int32)
                    target_positions_xyz = numpy.array(target_positions_xyz, dtype=numpy.int32)
                    distances = numpy.array(distances, dtype=numpy.int32)

                    for i in range(repeat_image):
                        image_with_links_list.append(
                            _ImageWithLinks(str(experiment.name), experiment.images, time_point, positions_xyz,
                                            target_positions_xyz, distances, linked))

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
                            [(future_possibility.x - position.x), (future_possibility.y - position.y),
                             (future_possibility.z - position.z)])

                        linked.append(False)

                        predicted_links.append((position, future_possibility))

        print(time_point)

        # read positions to numpy array
        max_size = 1000
        while len(positions_xyz) > 0:
            print(len(positions_xyz))
            if len(positions_xyz) > max_size:
                print('split link list')
            image_with_links_list.append(
                _ImageWithLinks(str(experiment.name), experiment.images, time_point,
                                numpy.array(positions_xyz[:max_size], dtype=numpy.int32),
                                numpy.array(target_positions_xyz[:max_size], dtype=numpy.int32),
                                numpy.array(distances[:max_size], dtype=numpy.int32),
                                linked[:max_size]))

            predicted_links_list.append(predicted_links[:max_size])

            positions_xyz = positions_xyz[max_size:]
            target_positions_xyz = target_positions_xyz[max_size:]
            distances = distances[max_size:]
            linked = linked[max_size:]
            predicted_links = predicted_links[max_size:]

    return image_with_links_list, predicted_links_list, possible_links


def inside_image(position: Position, offset: Images.offsets, image_shape: Tuple[int]):
    inside = False
    if position.x - offset.x < image_shape[2] and position.y - offset.y < image_shape[1] \
            and position.z - offset.z < image_shape[0]:
        inside = True

    if position.x - offset.x >= 0 and position.y - offset.y >= 0 and position.z - offset.z >= 0:
        inside = inside
    else:
        inside = False

    return inside


def remove_random_positions(experiment, rate=0.02):
    to_remove = []
    for pos in experiment.positions:
        if random.random() < rate:
            to_remove.append(pos)

    experiment.remove_positions(to_remove)

    return experiment
