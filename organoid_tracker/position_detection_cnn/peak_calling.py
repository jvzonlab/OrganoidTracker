from typing import Tuple

import numpy as np

# Helps peak calling
def _reconstruct_volume(multi_im: np.ndarray, mid_layers_nb: int) -> Tuple[np.ndarray, int]:
    """Reconstructs a volume so that the xy scale is roughly equal to the z scale. Returns the used scale."""
    # Make sure that
    if mid_layers_nb < 0:
        raise ValueError("negative number of mid layers")
    if mid_layers_nb == 0:
        return multi_im, 1  # No need to reconstruct anything

    out_img = np.zeros(
        (int(len(multi_im) + mid_layers_nb * (len(multi_im) - 1) + 2 * mid_layers_nb),) + multi_im[0].shape).astype(
        multi_im[0].dtype)

    layer_index = mid_layers_nb + 1
    orig_index = []

    for i in range(len(multi_im) - 1):

        for layer in range(mid_layers_nb + 1):
            t = float(layer) / (mid_layers_nb + 1)
            interpolate = ((1 - t) * (multi_im[i]).astype(float) + t * (multi_im[i + 1]).astype(float))

            out_img[layer_index] = interpolate

            if t == 0:
                orig_index.append(layer_index)
            layer_index += 1

    return out_img, mid_layers_nb + 1