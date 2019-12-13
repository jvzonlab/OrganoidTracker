import os
from typing import Optional, Tuple

import numpy
from matplotlib import pyplot
from tifffile import tifffile

from ai_track.core.experiment import Experiment
from ai_track.core.images import Images
from ai_track.core.position_collection import PositionCollection
from ai_track.imaging import cropper
from ai_track.linking_analysis import linking_markers
from ai_track.position_detection_cnn import predicter


def predict(experiment: Experiment, checkpoint_dir: str, out_dir: Optional[str] = None, split: bool = False,
            check_size_xyz_px: Tuple[int, int, int] = (16, 16, 3)):
    images = experiment.images
    positions = experiment.positions
    position_data = experiment.position_data
    check_size_x_px, check_size_y_px, check_size_z_px = check_size_xyz_px

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    output_array = numpy.zeros((check_size_z_px, check_size_y_px, check_size_x_px), dtype=numpy.float32)

    for time_point, prediction in predicter.predict_images(images, checkpoint_dir, split):
        print("Working on time point", time_point.time_point_number(), "...")

        # Save image if requested
        if out_dir is not None:
            image_name = "image_" + str(time_point.time_point_number())
            tifffile.imsave(os.path.join(out_dir, '{}.tif'.format(image_name)), prediction, compress=9)

        image_offset = images.offsets.of_time_point(time_point)
        existing_positions = positions.of_time_point(time_point)
        for existing_position in existing_positions:
            x_start = int(existing_position.x - image_offset.x - (check_size_x_px / 2))
            y_start = int(existing_position.y - image_offset.y - (check_size_y_px / 2))
            z_start = int(existing_position.z - image_offset.z - (check_size_z_px / 2))

            cropper.crop_3d(prediction, x_start, y_start, z_start, output_array)
            value = float(output_array.mean())
            if value >= 0.01:
                linking_markers.set_mother_score(position_data, existing_position, value)


