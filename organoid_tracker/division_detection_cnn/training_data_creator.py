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

from typing import List, Iterable

import numpy
from numpy import ndarray
import random

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Images
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
def create_image_with_divisions_list(experiments: Iterable[Experiment], non_divisions_per_frame=50, division_multiplier = 10):
    image_with_divisions_list = []
    for experiment in experiments:
        # read a complete experiment and identify the positions where a division takes place
        div_positions = list(cell_division_finder.find_mothers(experiment.links))

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
            positions_xyz = list()
            positions_xyz_div = list()

            print(time_point)

            # list dividing positions at the relevant time_points
            for div_time_point, div_position in zip(div_time_points, div_positions):
                if div_time_point.__eq__(time_point):
                    positions_xyz_div.append(
                        [div_position.x - offset.x, div_position.y - offset.y, div_position.z - offset.z])

            for position in positions:
                divide = False
                inside = False

                # check if the cell divides
                for position_div in positions_xyz_div:
                    if position.__eq__(position_div):
                        divide = True

                # check if position is inside the frame
                if position.x - offset.x < image_shape[2] and position.y - offset.y < image_shape[
                    1] and position.z - offset.z < image_shape[0]:
                    inside = True

                if position.x - offset.x >= 0 and position.y - offset.y >= 0 and position.z - offset.z >= 0:
                    inside = inside

                # add only non-dividing cells to this list
                if not divide and inside:
                    if not inside:
                        print('position out of frame')

                    positions_xyz.append([position.x - offset.x, position.y - offset.y, position.z - offset.z])

            # Add (multiple copies of) dividing cells and sample of non-dividing cells together
            if len(positions_xyz) > non_divisions_per_frame:
                positions_xyz = random.sample(positions_xyz, non_divisions_per_frame) + positions_xyz_div*division_multiplier
            else:
                positions_xyz = positions_xyz + positions_xyz_div * division_multiplier

            # make list into array
            positions_xyz = numpy.array(positions_xyz, dtype=numpy.int32)

            # make list with division info
            dividing = [False] * non_divisions_per_frame + [True] * len(positions_xyz_div*division_multiplier)

            # if there are divisions in the frame add it to the list
            if len(positions_xyz_div) > 0:
                image_with_divisions_list.append(
                    _ImageWithDivisions(str(experiment.name), experiment.images, time_point, positions_xyz, dividing))

    return image_with_divisions_list

# Create ImageWithPositions list for which to predict division status
def create_image_with_positions_list(experiment: Experiment):
    image_with_positions_list = []
    positions_per_frame_list = []

    for time_point in experiment.positions.time_points():
        # read a single time point
        print(time_point)
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
