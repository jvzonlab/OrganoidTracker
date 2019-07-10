from typing import List, Tuple, Optional

import cv2
import networkx
import numpy
from networkx import Graph
from numpy import ndarray

from ai_track.core.ellipse import EllipseStack, Ellipse

ELLIPSE_SHRINK_PIXELS = 5


def _max(a: Optional[int], b: Optional[int]) -> int:
    if a is None:
        return b
    if b is None:
        return a
    return a if a > b else b


def _min(a: Optional[int], b: Optional[int]) -> int:
    if a is None:
        return b
    if b is None:
        return a
    return a if a < b else b


class TaggedEllipseStack:
    """A stack of ellipses is tagged here with a numeric id, which is useful to keep track from which watershed area an
    ellipse came from."""
    _stack: EllipseStack
    _tag: int

    def __init__(self, stack: EllipseStack, tag: int):
        self._stack = stack
        self._tag = tag

    def get_tag(self) -> int:
        return self._tag

    def get_stack(self) -> EllipseStack:
        return self._stack


class EllipseCluster:
    """Multiple stacks of ellipses that are so close to each other that a Gaussian mixture model is necessary."""

    _stacks: List[TaggedEllipseStack]

    def __init__(self, stacks: List[TaggedEllipseStack]):
        self._stacks = list(stacks)

    def get_image_for_fit(self, original_image: ndarray, padding_xy: int = 2 + ELLIPSE_SHRINK_PIXELS, padding_z: int = 1
                          ) -> Tuple[int, int, int, Optional[ndarray]]:
        """Gets all pixels in the original image that belong to the given cell(s). These pixels can then be fitted to a
        Gaussian."""
        min_x, min_y, min_z, max_x, max_y, max_z = self._get_bounds()
        if min_x is None:
            return 0, 0, 0, None
        min_x = max(0, min_x - padding_xy)
        min_y = max(0, min_y - padding_xy)
        min_z = max(0, min_z - padding_z)
        max_x = min(original_image.shape[2], max_x + padding_xy + 1)
        max_y = min(original_image.shape[1], max_y + padding_xy + 1)
        max_z = min(original_image.shape[0], max_z + padding_z + 1)

        dx, dy, dz= max_x - min_x, max_y - min_y, max_z - min_z
        threshold_image = numpy.zeros((dz, dy, dx), dtype=numpy.uint8)
        for stack in self._stacks:
            stack.get_stack().draw_to_image(threshold_image, 255, -min_x, -min_y, -min_z, filled=True)
        original_image_cropped = numpy.copy(original_image[min_z:max_z, min_y:max_y, min_x:max_x])
        if original_image_cropped.shape != threshold_image.shape:
            raise ValueError("Error in shape")
        original_image_cropped &= threshold_image

        return min_x, min_y, min_z, original_image_cropped

    def _get_bounds(self):
        min_x, min_y, min_z = None, None, None
        max_x, max_y, max_z = None, None, None
        for stack in self._stacks:
            s_min_x, s_min_y, s_min_z, s_max_x, s_max_y, s_max_z = stack.get_stack().get_rectangular_bounds()
            min_x = _min(s_min_x, min_x)
            min_y = _min(s_min_y, min_y)
            min_z = _min(s_min_z, min_z)
            max_x = _max(s_max_x, max_x)
            max_y = _max(s_max_y, max_y)
            max_z = _max(s_max_z, max_z)
        return min_x, min_y, min_z, max_x, max_y, max_z

    def draw_to_image(self, out: ndarray, color, filled=False):
        for stack in self._stacks:
            stack.get_stack().draw_to_image(out, color, filled=filled)

    def get_tags(self) -> List[int]:
        """Gets the tags of all stacks. The returned order matches the order of self.guess_gaussians(...)"""
        return [stack.get_tag() for stack in self._stacks]

    def __repr__(self) -> str:
        return f"<EllipseCluster({len(self._stacks)} ellipse stacks)>"


def get_ellipse_stacks_from_watershed(watershed: ndarray) -> List[TaggedEllipseStack]:
    """Gets ellipse stacks from a watershed image. The stack tagged as nr. 0 is constructed from label 1, stack 1 from
    label 2, etc."""
    buffer = numpy.empty_like(watershed, dtype=numpy.uint8)
    ellipse_stacks = []
    for i in range(1, watershed.max() + 1):   # + 1 to ensure inclusive range
        ellipse_stack = []
        buffer.fill(0)
        buffer[watershed == i] = 255
        for z in range(buffer.shape[0]):
            contour_image, contours, hierarchy = cv2.findContours(buffer[z], cv2.RETR_LIST, 2)
            contour_index, area = _find_contour_with_largest_area(contours)
            if contour_index == -1:
                ellipse_stack.append(None)
                continue  # No contours found
            convex_contour = cv2.convexHull(contours[contour_index])
            if len(convex_contour) < 5 or area < 10 * 10:
                ellipse_stack.append(None)
                continue  # Contour or area too small for proper fit
            ellipse_pos, ellipse_size, ellipse_angle = cv2.fitEllipse(convex_contour)
            if ellipse_size[0] <= ELLIPSE_SHRINK_PIXELS or ellipse_size[1] <= ELLIPSE_SHRINK_PIXELS:
                ellipse_stack.append(None)
                continue  # Ellipse is too small
            ellipse_stack.append(Ellipse(ellipse_pos[0], ellipse_pos[1], ellipse_size[0] - ELLIPSE_SHRINK_PIXELS,
                                         ellipse_size[1] - ELLIPSE_SHRINK_PIXELS, ellipse_angle))
        ellipse_stacks.append(TaggedEllipseStack(EllipseStack(ellipse_stack), tag=i - 1))
    return ellipse_stacks


def find_overlapping_stacks(stacks: List[TaggedEllipseStack]) -> List[EllipseCluster]:
    """Finds all overlapping stacks, and returns those grouped in clusters. If a stack does not overlap with any other
    stack, it is returned as a "cluster" of one stack."""
    cell_network = Graph()
    for stack in stacks:
        cell_network.add_node(stack)
        for other_stack in stacks:
            if other_stack is stack:
                continue  # Ignore self-overlapping
            if other_stack not in cell_network:
                continue  # To be processed later
            if stack.get_stack().intersects(other_stack.get_stack()):
                cell_network.add_edge(stack, other_stack)

    clusters = []
    for cluster in networkx.connected_components(cell_network):
        clusters.append(EllipseCluster(cluster))
    return clusters


def _find_contour_with_largest_area(contours) -> Tuple[int, float]:
    highest_area = 0
    index_with_highest_area = -1
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > highest_area:
            highest_area = area
            index_with_highest_area = i
    return index_with_highest_area, highest_area
