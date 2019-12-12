import os
from typing import Optional

import mahotas
import numpy
from skimage.feature import peak_local_max
from tifffile import tifffile

from ai_track.core.images import Images
from ai_track.core.position import Position
from ai_track.core.position_collection import PositionCollection
from ai_track.position_detection_cnn import predicter
from ai_track.util import bits


def predict(images: Images, checkpoint_dir: str, out_dir: Optional[str] = None, split: bool = False,
            smooth_stdev: int = 0, predictions_threshold: float = 0.1) -> PositionCollection:
    """Applies the neural network to the images, detects the peaks in the output and returns those."""
    all_positions = PositionCollection()

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    for time_point, prediction in predicter.predict_images(images, checkpoint_dir, split):
        print("Working on time point", time_point.time_point_number(), "...")
        image_offset = images.offsets.of_time_point(time_point)

        # Apply smoothing
        if smooth_stdev > 0:
            prediction = mahotas.gaussian_filter(prediction, smooth_stdev)
        prediction = bits.image_to_8bit(prediction)  # Scipy peak finder doesn't accept the data format from mahotas

        # Save image if requested
        if out_dir is not None:
            image_name = "image_" + str(time_point.time_point_number())
            tifffile.imsave(os.path.join(out_dir, '{}.tif'.format(image_name)), prediction, compress=9)

        # Find local maxima
        coordinates = numpy.array(peak_local_max(prediction, min_distance=2, threshold_abs=predictions_threshold * 255,
                                                 exclude_border=False))
        for i in range(1, len(coordinates)):
            coordinate = coordinates[i]
            position = Position(coordinate[2], coordinate[1], coordinate[0], time_point=time_point) + image_offset
            if not _has_neighbor(position, all_positions):
                # If the peak is flat, then ALL pixels with the max value get returned as peaks. We only want one.
                all_positions.add(position)

    return all_positions


def _has_neighbor(position: Position, all_positions: PositionCollection) -> bool:
    """Checks if there's another position exactly 1 integer coord further."""
    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                if all_positions.contains_position(position.with_offset(dx, dy, dz)):
                    return True
    return False
