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
from organoid_tracker.neural_network.position_detection_cnn import _ImageWithPositions


# ImageWithDivisions extends ImageWithositions to include division data
class _ImageWithDivisions(_ImageWithPositions):
    _dividing = List[bool]

    def __init__(self, experiment_name: str, images: Images, time_point: TimePoint, xyz_positions: ndarray,
                 dividing: List[bool]):
        # dividing: List of booleans that indictae if a cell is about to divide (TRUE) or not (FALSE)
        super().__init__(experiment_name, images, time_point, xyz_positions)
        self._dividing = dividing


# Creates list of ImagesWithDivisions from experiments
def create_image_with_divisions_list(experiments: Iterable[Experiment], division_multiplier=10, loose_end_multiplier=1,
                                     counter_examples_per_div=0.25, window=(3, 2), loose_end_window=1,
                                     full_window=False, exclusion_window=None):
    # def create_image_with_divisions_list(experiments: Iterable[Experiment], division_multiplier=10, loose_end_multiplier=0,
    # counter_examples_per_div=8, window=(0, 0), loose_end_window = 1,
    # full_window=False, exclusion_window=0): (this gives the settings of an 'unadapted model' in the organoidtracker 2.0 publication)
    """"Creates training data set. Allows upsampling of dividing and dying cells (loose ends). If full_window is TRUE
    all cells within the window around cell division are deemed dividing, otherwise only the anaphase is classified as division.
    The exclusion window allows the exclusion of difficult cases just outside the window if full_window=TRUE."""

    # if the exclusion_window is not set, use default of 2 if needed
    if full_window:
        exclusion_window = 2
    else:
        exclusion_window = 0

    image_with_divisions_list = []
    for experiment in experiments:

        # read a complete experiment and identify the positions where a division takes place
        links = experiment.links
        div_positions = cell_division_finder.find_mothers(links, exclude_multipolar=False)

        div_positions_plus_window = set()
        div_positions_exclusion_window = set()

        for div_pos in div_positions:
            children = links.find_futures(div_pos)

            for child in children:
                futures = list(links.iterate_to_future(child))
                div_positions_plus_window = div_positions_plus_window.union(set(futures[:window[1]]))

                # remove potentially difficult cases when training to recognize a the full division window
                if (exclusion_window != 0):
                    div_positions_exclusion_window = div_positions_exclusion_window.union(
                        set(futures[window[1]:min(window[1] + exclusion_window, len(futures))]))

            pasts = list(links.iterate_to_past(div_pos))
            div_positions_plus_window = div_positions_plus_window.union(set(pasts[:window[0] + 1]))

            # remove potentially difficult cases when training to recognize a the full division window
            if (exclusion_window != 0):
                div_positions_exclusion_window = div_positions_exclusion_window.union(
                    set(pasts[(window[0] + 1):min(window[0] + 1 + exclusion_window, len(pasts))]))

        # what are the dividing cells? all cells in the window?
        if full_window:
            div_positions = div_positions_plus_window

        # find loose ends, possibly extruding cells
        loose_ends = list(experiment.links.find_disappeared_positions(
            time_point_number_to_ignore=experiment.last_time_point_number()))
        end_positions_plus_window = set()

        if loose_end_multiplier > 0:
            for end_pos in loose_ends:
                pasts = list(links.iterate_to_past(end_pos))
                end_positions_plus_window = end_positions_plus_window.union(
                    set(pasts[:min(loose_end_window + 1, len(pasts))]))

        # get the time points where divisions happen
        div_time_points = []
        for pos in div_positions:
            div_time_points.append(pos.time_point())

        for time_point in experiment.positions.time_points():

            positions = list(experiment.positions.of_time_point(time_point))
            random.shuffle(positions)

            offset = experiment.images.offsets.of_time_point(time_point)
            image_shape = experiment._images.get_image_stack(time_point).shape

            # read positions to numpy array
            positions_xyz = []
            dividing = []

            non_division_counter = 0
            division_counter = 1  # start with one to so there is way to select all cells

            for position in positions:
                divide = False
                repeat = 1

                # check if the cell divides
                if position in div_positions:
                    divide = True
                    # dividing cells are represented 1:2 vs children and the non-dividing cell in the earlier timepoint
                    # (so 1:1 in the end)
                    repeat = 1  # max(window[0] + 2*window[1], 1)

                # is cell dividing?
                if position in div_positions_plus_window:
                    repeat = division_multiplier * repeat
                    division_counter = division_counter + 1
                # is it dying?
                elif (position not in div_positions_exclusion_window) and (position in end_positions_plus_window):
                    repeat = loose_end_multiplier
                # do we add it an anyway?
                elif position not in div_positions_exclusion_window:
                    # non_division_counter = non_division_counter + 1
                    if non_division_counter >= (counter_examples_per_div * division_multiplier * division_counter):
                        repeat = 0
                    else:
                        non_division_counter = non_division_counter + 1
                else:
                    repeat = 0

                # check if position is inside the frame
                if inside_image(position, offset, image_shape):
                    for i in range(repeat):
                        positions_xyz.append([position.x - offset.x, position.y - offset.y, position.z - offset.z])
                        dividing.append(divide)
                else:
                    print('position out of frame')
                    print(str(experiment.name))
                    print(time_point)
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
            if position.x - offset.x < image_shape[2] and position.y - offset.y < image_shape[
                1] and position.z - offset.z < image_shape[0]:
                inside = True
            else:
                print('outside the image')
                print(offset.z)
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
    if position.x - offset.x < image_shape[2] and position.y - offset.y < image_shape[1] \
            and position.z - offset.z < image_shape[0]:
        inside = True

    if position.x - offset.x >= 0 and position.y - offset.y >= 0 and position.z - offset.z >= 0:
        inside = inside
    else:
        inside = False

    return inside
