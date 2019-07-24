from typing import List, Tuple, Optional, Set, Iterable

import mahotas
import numpy
from numpy import ndarray

class LabeledCluster:
    """Multiple stacks of ellipses that are so close to each other that a Gaussian mixture model is necessary."""

    _labels: Set[int]

    def __init__(self):
        self._labels = set()

    def get_tags(self) -> List[int]:
        """Gets the tags of all stacks. The returned order matches the order of self.guess_gaussians(...)"""
        return list(self._labels)

    def __repr__(self) -> str:
        return f"<LabeledCluster({repr(self._labels)})>"


def get_clusters_from_labeled_image(watershed_image: ndarray, positions_zyx_list: ndarray, erode_passes: int = 2) -> List[LabeledCluster]:
    """Gets positions clusters from a watershed image. The tags from all returned clusters are the indices in the
    positions_zyx_list."""
    # Use a watershed to find connected (overlapping) cells
    threshold = numpy.zeros_like(watershed_image, dtype=numpy.uint16)
    threshold[watershed_image != 0] = 255
    for i in range(erode_passes):
        threshold = mahotas.morph.erode(threshold)
    connected_components_image, count = mahotas.label(threshold)
    connected_components_image = mahotas.cwatershed(numpy.zeros_like(threshold), connected_components_image )

    #connected_components_image[:, 0, 0] = count + 1
    #popup_3d_image(connected_components_image, "clusters", cm.jet)

    # Divide positions into clusters
    clusters = [LabeledCluster() for i in range(count)]  # Create N empty lists
    for i in range(positions_zyx_list.shape[0]):
        position_zyx = positions_zyx_list[i]
        if numpy.isnan(position_zyx[0]):
            continue  # No positions for that index in the watershed found
        cluster_index = connected_components_image[int(position_zyx[0]), int(position_zyx[1]), int(position_zyx[2])]
        if cluster_index == 0:
            print("Outside regions:", position_zyx)
        clusters[cluster_index]._labels.add(i)

    # Return all non-empty clusters
    return [cluster for cluster in clusters if len(cluster._labels) > 0]
