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

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.images import Images
from organoid_tracker.core.position import Position


class _ImageWithPositions:
    _time_point: TimePoint
    _images: Images
    xyz_positions: ndarray
    experiment_name: str

    def __init__(self, experiment_name: str, images: Images, time_point: TimePoint, xyz_positions: ndarray):
        # xyz positions: 2D numpy integer array of cell nucleus positions: [ [x,y,z], [x,y,z], ...]
        self.experiment_name = experiment_name
        self._time_point = time_point
        self._images = images
        self.xyz_positions = xyz_positions

    def __str__(self) -> str:
        return f"{self.experiment_name} t{self._time_point.time_point_number()}"

    def load_image(self, dt: int = 0) -> Optional[ndarray]:
        time_point = TimePoint(self._time_point.time_point_number() + dt)
        return self._images.get_image_stack(time_point)

    def get_image_size_zyx(self, dt: int = 0) -> Tuple[int, int, int]:
        """Gets the shape of the image. This method tries to avoid loading the actual image data to find it out."""
        size = self._images.image_loader().get_image_size_zyx()
        if size is None:
            return self.load_image(dt).shape
        return size

    # loads images in a time window
    def load_image_time_stack(self, time_window: List[int] = [0, 0]) -> Optional[ndarray]:

        center_image = self.load_image()
        offset_ref = self._images.offsets.of_time_point(self._time_point)
        image_shape_ref = center_image.shape

        def aligner(image: ndarray, image_ref: ndarray, offset: Position, offset_ref: Position):
            shift = offset - offset_ref

            shift.x = round(shift.x)
            shift.y = round(shift.y)
            shift.z = round(shift.z)

            # shift images according to offsets
            image = numpy.roll(image, shift=(shift.z, shift.y, shift.x), axis=(0, 1, 2))

            # if in region information at a timepoint is unknown due to shifts the t=0 image is copied
            if shift.x < 0:
                image[:, :, shift.x:] = image_ref[:, :, shift.x:]
            else:
                image[:, :, :shift.x] = image_ref[:, :, :shift.x]

            if shift.y < 0:
                image[:, shift.y:, :] = image_ref[:, shift.y:, :]
            else:
                image[:, :shift.y, :] = image_ref[:, :shift.y, :]

            if shift.z < 0:
                image[shift.z:, :, :] = image_ref[shift.z:, :, :]
            else:
                image[:shift.z, :, :] = image_ref[:shift.z, :, :]

            return image

        images = list()
        # records at which timepoints images were available
        image_dt = list()

        frames = range(time_window[0], time_window[1]+1)

        for dt in frames:
            image = self.load_image(dt)

            if image is not None and image.shape == image_shape_ref:
                time_point = TimePoint(self._time_point.time_point_number() + dt)
                offset = self._images.offsets.of_time_point(time_point)

                image = aligner(image, center_image, offset, offset_ref)
                images.append(image)

                image_dt.append(dt)

        # pads timestack if images are missing
        images_padded = list()
        for dt in frames:
            if dt < numpy.min(image_dt):
                images_padded.append(images[0])

        images_padded = images_padded + images

        for dt in frames:
            if dt > numpy.max(image_dt):
                images_padded.append(images[-1])

        stack = numpy.stack(images_padded, axis=-1)

        return stack

    def create_labels(self, image_size_zyx: Tuple[int, int, int]):
        sub_data_xyz = self.xyz_positions
        markers = numpy.zeros(image_size_zyx).astype(numpy.float32)

        max_x = image_size_zyx[2]
        max_y = image_size_zyx[1]
        max_z = image_size_zyx[0]

        x = sub_data_xyz[:, 0]
        y = sub_data_xyz[:, 1]
        z = sub_data_xyz[:, 2]-1

        in_range = numpy.where((x >= max_x) + (y >= max_y) + (z >= max_z) == 0)

        x = tuple(x[in_range])
        y = tuple(y[in_range])
        z = tuple(z[in_range])
        values = [1] * len(z)

        markers[z, y, x] = values

        return markers


def create_image_with_positions_list(experiments: Iterable[Experiment]):
    image_with_positions_list = []
    for experiment in experiments:
        # read a complete experiment

        for time_point in experiment.positions.time_points():
            # read a single time point
            positions = experiment.positions.of_time_point(time_point)
            offset = experiment.images.offsets.of_time_point(time_point)

            # read positions to numpy array
            positions_xyz = list()
            for position in positions:
                positions_xyz.append([position.x - offset.x, position.y - offset.y, position.z - offset.z])
            positions_xyz = numpy.array(positions_xyz, dtype=numpy.int32)

            image_with_positions_list.append(
                _ImageWithPositions(str(experiment.name), experiment.images, time_point, positions_xyz))

    return image_with_positions_list


def create_image_list_without_positions(experiment: Experiment):
    image_list = []

    for time_point in experiment.time_points():
        image_list.append(
            _ImageWithPositions(str(experiment.name), experiment.images, time_point, numpy.empty((0, 3), dtype=numpy.float32)))

    return image_list


