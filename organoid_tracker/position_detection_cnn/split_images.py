""""Helps to split images in smaller parts and the reconstructs them"""
from typing import List, Union

import numpy as np


# gives a list of the corner coordinates (top-left) of the splits
def corners_split(image_shape: Union[np.ndarray, List[int]], patch_shape: Union[np.ndarray, List[int]]
                  ) -> List[np.ndarray]:
    corners = list([np.zeros(3, dtype=int)])

    for dim in range(3):

        if image_shape[dim] > patch_shape[dim]:
            # minimum of patches needed
            num_corners = (image_shape[dim] - patch_shape[dim]) // patch_shape[dim] + 2

            # equal spacing between corners
            spacing = round((image_shape[dim] - patch_shape[dim]) / (num_corners - 1))

            coords = np.arange(num_corners, dtype=int) * spacing
        else:
            coords = [np.array(0)]

        corners_old = list(corners)
        corners = []

        # clever/hacky way to get all combinations of x-y-z coner coordinates
        for coord in coords:
            for corner in corners_old:
                new_corner = np.array(corner)
                new_corner[dim] = coord
                corners.append(new_corner)

    return corners


def reconstruction(image_batch, corners, buffer, image_shape, patch_shape):
    # add channel dimension and create empty volume
    final_image = np.zeros(image_shape + [image_batch.shape[4]], dtype=np.float32)
    # records if volume is filled by a patch
    filled = np.zeros(image_shape + [image_batch.shape[4]], dtype=np.uint8)

    for i, corner in zip(range(len(corners)), corners):
        input_data = image_batch[i,]
        # remove buffer zone
        input_data = input_data[buffer[0, 0]: patch_shape[0]+buffer[0, 0],
                     buffer[1, 0]: patch_shape[1]+buffer[1, 0],
                     buffer[2, 0]: patch_shape[2]+buffer[2, 0], :]

        # data to replace/combine with
        replace_data = final_image[corner[0]: corner[0] + patch_shape[0],
                       corner[1]: corner[1] + patch_shape[1],
                       corner[2]: corner[2] + patch_shape[2], :]

        # ensure proper size
        input_data = input_data[:replace_data.shape[0],
                                :replace_data.shape[1],
                                :replace_data.shape[2], :]

        # is this place already filled?
        replace_filled = filled[corner[0]: corner[0] + patch_shape[0],
                         corner[1]: corner[1] + patch_shape[1],
                         corner[2]: corner[2] + patch_shape[2], :]

        final_image[corner[0]: corner[0] + patch_shape[0],
        corner[1]: corner[1] + patch_shape[1],
        corner[2]: corner[2] + patch_shape[2], :] = np.where(replace_filled == 0,
                                                             input_data,
                                                             np.mean([input_data, replace_data], axis=0))
        filled[corner[0]: corner[0] + patch_shape[0],
        corner[1]: corner[1] + patch_shape[1],
        corner[2]: corner[2] + patch_shape[2], :] = 1

    return final_image
