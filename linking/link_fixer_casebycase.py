# The following rules can be used:
# * Every cell must have one or two cells in the next image
# * Every cell must have exactly one cell in the previous image

from typing import Iterable, Set, Optional, Tuple

import numpy
from networkx import Graph
from numpy import ndarray
import cv2
import math

from imaging import Particle, Experiment, errors, normalized_image
from linking.link_fixer import downgrade_edges_pointing_to_past, find_preferred_links, find_preferred_past_particle, \
    find_future_particles, remove_error, with_only_the_preferred_edges, get_2d_image, fix_no_future_particle, \
    get_closest_particle_having_a_sister
from linking_analysis import logical_tests


def prune_links(experiment: Experiment, graph: Graph, mitotic_radius: int) -> Graph:
    """Takes a graph with all possible edges between cells, and returns a graph with only the most likely edges.
    mitotic_radius is the radius used to detect whether a cell is undergoing mitosis (i.e. it will have divided itself
    into two in the next time_point). For non-mitotic cells, it must fall entirely within the cell, for mitotic cells it must
    fall partly outside the cell.
    """
    for i in range(2):
        [fix_no_future_particle(graph, particle) for particle in graph.nodes()]
        [_fix_cell_divisions(experiment, graph, particle, mitotic_radius) for particle in graph.nodes()]

    graph = with_only_the_preferred_edges(graph)
    logical_tests.apply(experiment, graph)
    return graph


def _fix_cell_divisions(experiment: Experiment, graph: Graph, particle: Particle, mitotic_radius: int):
    future_particles = find_future_particles(graph, particle)
    future_preferred_particles = find_preferred_links(graph, particle, future_particles)

    if len(future_particles) < 2 or len(future_preferred_particles) == 0:
        return # Surely not a mother cell

    if len(future_preferred_particles) > 2:
        graph.add_node(particle, error=errors.TOO_MANY_DAUGHTER_CELLS)
        return

    two_daughters = _get_two_daughters(graph, particle, future_preferred_particles, future_particles)
    if two_daughters is None:
        print("Cannot fix " + str(particle) + ", no other mother nearby")
        return

    # Daughter1 surely is in preferred_particles, but maybe daughter2 not yet. If so, we might need to declare this cell
    # as a mother, and "undeclare" another cell from being one
    daughter2 = two_daughters[1]
    if daughter2 in future_preferred_particles:
        print("No need to fix " + str(particle))
        return  # Nothing to fix
    current_mother_of_daughter2 = find_preferred_past_particle(graph, daughter2)
    children_of_current_mother_of_daughter2 = list(find_preferred_links(graph, current_mother_of_daughter2,
                                                   find_future_particles(graph, current_mother_of_daughter2)))
    if len(children_of_current_mother_of_daughter2) < 2:
        # The _get_two_daughters should have checked for this
        raise ValueError("No nearby mother available for " + str(particle))

    score = _cell_is_mother_likeliness(experiment, particle, two_daughters[0], two_daughters[1], mitotic_radius)
    current_parent_score = _cell_is_mother_likeliness(experiment, current_mother_of_daughter2,
                                                      children_of_current_mother_of_daughter2[0],
                                                      children_of_current_mother_of_daughter2[1], mitotic_radius)
    # Printing of warnings
    if abs(score - current_parent_score) <= 1:
        # Not sure
        if score > current_parent_score:
            graph.add_node(particle, error=errors.POTENTIALLY_NOT_A_MOTHER)
        else:
            graph.add_node(current_mother_of_daughter2, error=errors.POTENTIALLY_NOT_A_MOTHER)
    else:  # Remove any existing errors, they will be outdated
        remove_error(graph, particle)
        remove_error(graph, current_mother_of_daughter2)

    # Parent replacement
    if score > current_parent_score:
        # Replace parent
        print("Let " + str(particle) + " (score " + str(score) + ") replace " + str(
            current_mother_of_daughter2) + " (score " + str(current_parent_score) + ")")
        downgrade_edges_pointing_to_past(graph, daughter2)  # Removes old mother
        graph.add_edge(particle, daughter2, pref=True)
        return True
    else:
        print("Didn't let " + str(particle) + " (score " + str(score) + ") replace " + str(
            current_mother_of_daughter2) + " (score " + str(current_parent_score) + ")")
        return False


#
# Helper functions below
#


def _get_two_daughters(graph: Graph, mother: Particle, already_declared_as_daughter: Set[Particle],
                       all_future_cells: Set[Particle]) -> Optional[Tuple[Particle, Particle]]:
    """Gets a list with two daughter cells at positions 0 and 1. First, particles from te preferred lists are chosen,
    then the nearest from the other list are chosen. The order of the cells in the resulting list is not defined.
    """
    result = list(already_declared_as_daughter)
    in_consideration = set(all_future_cells)
    for preferred_particle in already_declared_as_daughter:
        in_consideration.remove(preferred_particle)

    while len(result) < 2:
        nearest = get_closest_particle_having_a_sister(graph, in_consideration, mother)
        if nearest is None:
            return None # Simply not enough cells provided
        result.append(nearest)
        in_consideration.remove(nearest)

    return (result[0], result[1])


def _cell_is_mother_likeliness(experiment: Experiment, mother: Particle, daughter1: Particle,
                               daughter2: Particle, mitotic_radius: int = 2):

    mother_image_stack = experiment.get_time_point(mother.time_point_number()).load_images()
    daughter1_image_stack = experiment.get_time_point(daughter1.time_point_number()).load_images()
    mother_image = mother_image_stack[int(mother.z)]
    mother_image_next = daughter1_image_stack[int(mother.z)]
    daughter1_image = daughter1_image_stack[int(daughter1.z)]
    daughter2_image = get_2d_image(experiment, daughter2)
    daughter1_image_prev = mother_image_stack[int(daughter1.z)]
    daughter2_image_prev = mother_image_stack[int(daughter2.z)]

    try:
        mother_intensities = normalized_image.get_square(mother_image, mother.x, mother.y, mitotic_radius)
        mother_intensities_next = normalized_image.get_square(mother_image_next, mother.x, mother.y, mitotic_radius)
        daughter1_intensities = normalized_image.get_square(daughter1_image, daughter1.x, daughter1.y, mitotic_radius)
        daughter2_intensities = normalized_image.get_square(daughter2_image, daughter2.x, daughter2.y, mitotic_radius)
        daughter1_intensities_prev = normalized_image.get_square(daughter1_image_prev, daughter1.x, daughter1.y,
                                                                 mitotic_radius)
        daughter2_intensities_prev = normalized_image.get_square(daughter2_image_prev, daughter2.x, daughter2.y,
                                                                 mitotic_radius)

        score = 0
        score += score_mother_intensities(mother_intensities, mother_intensities_next)
        score += score_daughter_intensities(daughter1_intensities, daughter2_intensities,
                                            daughter1_intensities_prev, daughter2_intensities_prev)
        score += score_daughter_positions(mother, daughter1, daughter2)
        score += score_mother_shape(mother, mother_image)
        return score
    except IndexError:
        return 0


def score_daughter_positions(mother: Particle, daughter1: Particle, daughter2: Particle) -> int:
    m_d1_distance = mother.distance_squared(daughter1)
    m_d2_distance = mother.distance_squared(daughter2)
    shorter_distance = m_d1_distance if m_d1_distance < m_d2_distance else m_d2_distance
    longer_distance = m_d1_distance if m_d1_distance > m_d2_distance else m_d2_distance
    if shorter_distance * (6 ** 2) < longer_distance:
        return -2
    return 0


def score_daughter_intensities(daughter1_intensities: ndarray, daughter2_intensities: ndarray,
                               daughter1_intensities_prev: ndarray, daughter2_intensities_prev: ndarray):
    """Daughter cells must have almost the same intensity"""
    daughter1_average = numpy.average(daughter1_intensities)
    daughter2_average = numpy.average(daughter2_intensities)
    daughter1_average_prev = numpy.average(daughter1_intensities_prev)
    daughter2_average_prev = numpy.average(daughter2_intensities_prev)

    # Daughter cells must have almost the same intensity
    score = 0
    score -= abs(daughter1_average - daughter2_average) / 2
    if daughter1_average / (daughter1_average_prev + 0.0001) > 2:
        score += 1
    if daughter2_average / (daughter2_average_prev + 0.0001) > 2:
        score += 1
    return score


def score_mother_intensities(mother_intensities: ndarray, mother_intensities_next: ndarray) -> float:
    """Mother cell must have high intensity """
    score = 0

    # Intensity and contrast
    min_value = numpy.min(mother_intensities)
    max_value = numpy.max(mother_intensities)
    if max_value > 0.7:
        score += 1  # The higher intensity, the better: the DNA is concentrated
    if max_value - min_value > 0.4:
        score += 0.5  # High contrast is also desirable, as there are parts where there is no DNA

    # Change of intensity (we use the max, as mothers often have both bright spots and darker spots near their center)
    max_value_next = numpy.max(mother_intensities_next)
    if max_value / (max_value_next + 0.0001) > 2: # +0.0001 protects against division by zero
        score += 1

    return score


def score_mother_shape(mother: Particle, full_image: ndarray, detection_radius = 16) -> int:
    """Returns a black-and-white image where white is particle and black is background, at least in theory."""

    # Zoom in on mother
    x = int(mother.x)
    y = int(mother.y)
    if x - detection_radius < 0 or y - detection_radius < 0 or x + detection_radius >= full_image.shape[1] \
            or y + detection_radius >= full_image.shape[0]:
        return 0  # Out of bounds
    image = full_image[y - detection_radius:y + detection_radius, x - detection_radius:x + detection_radius]
    image_8bit = cv2.convertScaleAbs(image, alpha=256 / image.max(), beta=0)

    # Crop to a circle
    image_circle = numpy.zeros_like(image_8bit)
    width = image_circle.shape[1]
    height = image_circle.shape[0]
    circle_size = min(width, height) - 3
    cv2.circle(image_circle, (int(width/2), int(height/2)), int(circle_size/2), 255, -1)
    cv2.bitwise_and(image_8bit, image_circle, image_8bit)

    # Find contour
    ret, thresholded_image = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_image, contours, hierarchy = cv2.findContours(thresholded_image, 1, 2)

    # Calculate the isoperimetric quotient of the largest area
    highest_area = 0
    isoperimetric_quotient = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > highest_area:
            highest_area = area
            perimeter = cv2.arcLength(contour, True)
            isoperimetric_quotient = 4 * math.pi * area / perimeter**2 if perimeter > 0 else 0

    if isoperimetric_quotient < 0.4:
        # Clear case of being a mother, give a bonus
        return 2
    # Just use a normal scoring system
    return 1 - isoperimetric_quotient
