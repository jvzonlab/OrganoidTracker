from typing import List, Tuple, Optional, Set, Iterable

import mahotas
import numpy
from matplotlib import cm
from numpy import ndarray

from organoid_tracker.core.position import Position
from organoid_tracker.util.mpl_helper import QUALITATIVE_COLORMAP
from organoid_tracker.visualizer.debug_image_visualizer import popup_3d_image


class LabeledCluster:
    """Multiple stacks of ellipses that are so close to each other that a Gaussian mixture model is necessary."""

    _cluster_index: int
    _labels: Set[int]

    def __init__(self, cluster_index: int):
        self._cluster_index = cluster_index
        self._labels = set()

    def get_tags(self) -> List[int]:
        """Gets the tags of all stacks. The returned order matches the order of self.guess_gaussians(...)"""
        return list(self._labels)

    @property
    def cluster_index(self) -> int:
        return self._cluster_index

    def __repr__(self) -> str:
        return f"<LabeledCluster({repr(self._labels)})>"


def get_clusters_from_labeled_image(watershed_image: ndarray, positions_list: List[Optional[Position]], erode_passes: int
                                    ) -> Tuple[List[LabeledCluster], ndarray]:
    """Gets positions clusters from a watershed image. The tags from all returned clusters are the indices in the
    positions_zyx_list. Returns the clusters and the image used for clustering. Note that that image is eroded
    erode_passes times."""
    # Use a watershed to find connected (overlapping) cells
    threshold = numpy.zeros_like(watershed_image, dtype=numpy.uint16)
    threshold[watershed_image != 0] = 255
    for i in range(erode_passes):
        threshold = mahotas.morph.erode(threshold)
    connected_components_image, count = mahotas.label(threshold)
    connected_components_image = mahotas.cwatershed(numpy.zeros_like(threshold), connected_components_image )

    # If you want to see the image:
    # connected_components_image[threshold == 0] = 0  # Makes areas outside threshold black
    # connected_components_image[:, 0, 0] = count + 1  # Makes colors scale the same at every layer
    # popup_3d_image(connected_components_image, "clusters", QUALITATIVE_COLORMAP)

    # Divide positions into clusters
    clusters = [LabeledCluster(i) for i in range(count + 1)]  # Create N empty lists
    for i in range(len(positions_list)):
        if i == 0:
            continue  # Position 0 is the background - it doesn't represent a particle
        position = positions_list[i]
        if position is None:
            continue  # No positions for that index in the watershed found
        cluster_index = connected_components_image[int(position.z), int(position.y), int(position.x)]
        if cluster_index == 0:
            print("Outside regions:", position)
        clusters[cluster_index]._labels.add(i)

    # Return all non-empty clusters
    return [cluster for cluster in clusters if len(cluster._labels) > 0], connected_components_image
