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

from typing import List, Iterable, Tuple

import numpy
from numpy import ndarray
import random

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Images
from organoid_tracker.core.position import Position
from organoid_tracker.linking import cell_division_finder
from organoid_tracker.position_detection_cnn.training_data_creator import _ImageWithPositions


# ImageWithDivisions extends ImageWithositions to include division data
class _ImageWithDivisions(_ImageWithPositions):
    _dividing = List[bool]

    def __init__(self, experiment_name: str, images: Images, time_point: TimePoint, xyz_positions: ndarray,
                 dividing: List[bool]):
        # dividing: List of booleans that indictae if a cell is about to divide (TRUE) or not (FALSE)
        super().__init__(experiment_name, images, time_point, xyz_positions)
        self._dividing = dividing


# Creates list of ImagesWithDivisions from experiments
def create_image_with_divisions_list(experiments: Iterable[Experiment], division_multiplier = 10):
    image_with_divisions_list = []
    for experiment in experiments:

        # read a complete experiment and identify the positions where a division takes place
        links = experiment.links
        div_positions = cell_division_finder.find_mothers(links)

        div_child_positions = div_positions.copy()

        for div_pos in div_positions:
            children = links.find_futures(div_pos)
            div_child_positions = div_child_positions.union(children)

            mothers_before = links.find_pasts(div_pos)
            if mothers_before:
                div_child_positions = div_child_positions.union(mothers_before)

        # get the time points where divisions happen
        div_time_points = []
        for pos in div_positions:
            div_time_points.append(pos.time_point())

        for time_point in experiment.positions.time_points():
            # read a single time point
            positions = experiment.positions.of_time_point(time_point)
            offset = experiment.images.offsets.of_time_point(time_point)
            image_shape = experiment._images.get_image_stack(time_point).shape

            # read positions to numpy array
            positions_xyz = []
            dividing = []

            print(time_point)

            for position in positions:
                divide = False
                repeat = 1

                # check if the cell divides
                if position in div_positions:
                    divide = True
                    # dividing cells are represented 3:1 vs children and the non-dividing cell in the earlier timepoint
                    # (so 1:1 in the end)
                    repeat = 3
                    print('division found')

                if position in div_child_positions:
                    repeat = division_multiplier * repeat

                # check if position is inside the frame
                if inside_image(position, offset, image_shape):
                    for i in range(repeat):
                        positions_xyz.append([position.x - offset.x, position.y - offset.y, position.z - offset.z])
                        dividing.append(divide)

                else:
                    print('position out of frame')
                    print(offset.z)
                    print(position.z)

            # make list into array
            positions_xyz = numpy.array(positions_xyz, dtype=numpy.int32)

            # if there are positions in the frame add it to the list
            if len(positions_xyz) > 0:
                image_with_divisions_list.append(
                    _ImageWithDivisions(str(experiment.name), experiment.images, time_point, positions_xyz, dividing))

    return image_with_divisions_list

# Create ImageWithPositions list for which to predict division status
def create_image_with_positions_list(experiment: Experiment):
    image_with_positions_list = []
    positions_per_frame_list = []

    for time_point in experiment.positions.time_points():
        # read a single time point
        positions = experiment.positions.of_time_point(time_point)
        offset = experiment.images.offsets.of_time_point(time_point)
        image_shape = experiment._images.get_image_stack(time_point).shape

        # read positions to numpy array
        positions_xyz = list()
        positions_list = list()

        for position in positions:

            # check if the position is inside the image
            inside = False
            if position.x - offset.x < image_shape[2] and position.y - offset.y < image_shape[1] and position.z - offset.z < image_shape[0]:
                inside = True
            else:
                print('outside the image')
                print(time_point)
                print(position.z)
                print(image_shape[0])

            if position.x - offset.x >= 0 and position.y - offset.y >= 0 and position.z - offset.z >= 0:
                inside = inside

            if inside:
                positions_xyz.append([position.x - offset.x, position.y - offset.y, position.z - offset.z])
                positions_list.append(position)

        # array with all positions for single timepoint
        positions_xyz = numpy.array(positions_xyz, dtype=numpy.int32)

        # Add ImageWithPositions for that time_point
        image_with_positions_list.append(
            _ImageWithPositions(str(experiment.name), experiment.images, time_point, positions_xyz))

        # add positions as list for single time_point
        positions_per_frame_list.append(positions_list)

    return image_with_positions_list, positions_per_frame_list


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