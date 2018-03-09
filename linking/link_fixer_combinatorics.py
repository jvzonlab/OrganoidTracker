import itertools
from typing import Set, Dict, List, Iterable, Tuple

import numpy
from networkx import Graph
from numpy import ndarray
import cv2
import math

import imaging
from imaging import Particle, Experiment, TimePoint, normalized_image
from linking.link_fixer import fix_no_future_particle, with_only_the_preferred_edges, find_future_particles, \
    find_preferred_links, downgrade_edges_pointing_to_past, get_2d_image, find_preferred_past_particle
from linking_analysis import logical_tests
from linking_analysis.mother_finder import Family


class SearchResult:
    mothers: Set[Particle]
    mothers_unsure: Set[Particle]
    daughters: Set[Particle]
    daughters_unsure: Set[Particle]

    def __init__(self):
        self.mothers = set()
        self.mothers_unsure = set()
        self.daughters = set()
        self.daughters_unsure = set()

    def __str__(self):
        return "Mothers: " + str(len(self.mothers)) + " (+" + str(len(self.mothers_unsure)) +" unsure)" \
               + " Daughters: " + str(len(self.daughters)) + " (+" + str(len(self.daughters_unsure)) +" unsure)"

    def add_daughter(self, particle: Particle):
        self.daughters.add(particle)

    def add_mother(self, particle: Particle):
        self.mothers.add(particle)

    def add_mother_unsure(self, particle: Particle):
        self.mothers_unsure.add(particle)

    def add_daughter_unsure(self, particle: Particle):
        self.daughters_unsure.add(particle)

    def find_likely_candidates(self, max_distance: int):
        # Grab all likely mothers and daughters
        mothers = set(self.mothers)
        daughters = set(self.daughters)

        # Grab all unlikely mothers within max_distance of a likely daughter
        max_distance_squared = max_distance ** 2
        for daughter in self.daughters:
            for mother in self.mothers_unsure:
                if daughter.distance_squared(mother) < max_distance_squared:
                    mothers.add(mother)

        # Grab all unlikely daughters within max_distance of a likely mother
        for mother in self.daughters:
            for daughter in self.daughters_unsure:
                if mother.distance_squared(daughter) < max_distance_squared:
                    daughters.add(daughter)

        return mothers, daughters

class ScoredFamily:
    """A family with a score attached. The higher the score, the higher the chance that this family actually exists."""
    family: Family
    score: float

    def __init__(self, family: Family, score: float):
        self.family = family
        self.score = score

    def __repr__(self):
        return "<ScoredFamily " + str(self.score) + " " + str(self.family) + ">"


def prune_links(experiment: Experiment, graph: Graph, detection_radius_small: int, detection_radius_large: int,
                max_distance_mother_daughter: int) -> Tuple[Graph, Set[Family]]:
    """Takes a graph with all possible edges between cells, and returns a graph with only the most likely edges.
    mitotic_radius is the radius used to detect whether a cell is undergoing mitosis (i.e. it will have divided itself
    into two in the next time_point). For non-mitotic cells, it must fall entirely within the cell, for mitotic cells it must
    fall partly outside the cell.
    """
    all_families = set()

    [fix_no_future_particle(graph, particle) for particle in graph.nodes()]
    for time_point_number in range(experiment.first_time_point_number(), experiment.last_time_point_number() + 1):
        time_point = experiment.get_time_point(time_point_number)
        scored_families = _fix_cell_divisions_for_time_point(experiment, graph, time_point, detection_radius_small,
                                                           detection_radius_large, max_distance_mother_daughter)
        for scored_family in scored_families:
            all_families.add(scored_family.family)

    #[fix_no_future_particle(graph, particle) for particle in graph.nodes()]

    graph = with_only_the_preferred_edges(graph)
    logical_tests.apply(experiment, graph)
    return graph, all_families


def _fix_cell_divisions_for_time_point(experiment: Experiment, graph: Graph, time_point: TimePoint,
                                       detection_radius_small: int, detection_radius_large: int,
                                       max_distance_mother_daughter: int) -> List[ScoredFamily]:
    print("Working on time point " + str(time_point.time_point_number()))

    # Load images
    tp_images = time_point.load_images()
    try:
        next_time_point = experiment.get_next_time_point(time_point)
    except KeyError:
        return []  # Last time point, cannot do anything
    next_tp_images = next_time_point.load_images()

    # Find existing cell divisions (nearest-neighbor linking)
    cell_divisions = _get_cell_divisions(time_point, graph)
    expected_cell_divisions = len(next_time_point.particles()) - len(time_point.particles())
    if expected_cell_divisions <= 0:
        return []  # No cell divisions here

    # Find potential mothers and daughters, find best combinations
    possible_mothers_and_daughters = _find_mothers_and_daughters(time_point, next_time_point, tp_images, next_tp_images,
                                                                 detection_radius_small, detection_radius_large)
    mothers, daughters = possible_mothers_and_daughters.find_likely_candidates(max_distance_mother_daughter)
    family_stack = _perform_combinatorics(experiment, graph, expected_cell_divisions, mothers, daughters,
                                          max_distance=max_distance_mother_daughter)
    print("Mothers: " + str(len(mothers)) + " (expected " + str(expected_cell_divisions)+ ")  Daughters: "
          + str(len(daughters)) + " Best combination: " + str(family_stack))

    # Apply
    _replace_divisions(graph, old_divisions=cell_divisions, new_divisions=family_stack)

    return family_stack


def _replace_divisions(graph: Graph, old_divisions: Set[Family], new_divisions: Iterable[ScoredFamily]):
    for new_division in new_divisions:
        new_family = new_division.family
        if new_family in old_divisions:
            # This division was already detected
            old_divisions.remove(new_family)
            continue

        print("Adding new family: " + str(new_family))
        for existing_connection in find_future_particles(graph, new_family.mother):
            graph.add_edge(new_family.mother, existing_connection, pref=False)
        for daughter in new_family.daughters:
            downgrade_edges_pointing_to_past(graph, daughter)
            graph.add_edge(new_family.mother, daughter, pref=True)


def _perform_combinatorics(experiment: Experiment, graph: Graph, mothers_pick_amount: int, mothers: Set[Particle],
                           daughters: Set[Particle], max_distance: int) -> List[ScoredFamily]:
    # Loop through all possible combinations of N mothers, and picks the best combination of mothers and daughters
    nearby_daughters = dict()
    for mother in mothers:
        nearby_daughters[mother] = imaging.get_closest_n_particles(daughters, mother, 4, max_distance=max_distance)


    best_family_stack = []
    best_family_stack_score = 0
    for picked_mothers in itertools.combinations(mothers, mothers_pick_amount):
        for family_stack in _scan(dict(), experiment, graph, picked_mothers, 0, nearby_daughters):
            if family_stack[0].family.mother.time_point_number() == 48:
                print(family_stack)
            score = _score_stack(family_stack)
            if score > best_family_stack_score:
                best_family_stack = family_stack
                best_family_stack_score = score
    return best_family_stack


def _scan(cache: Dict[Family, ScoredFamily], experiment: Experiment, graph: Graph, mothers: List[Particle], mother_pos: int,
          nearby_daughters_dict: Dict[Particle, Set[Particle]], family_stack: List[ScoredFamily] = []):
    mother = mothers[mother_pos]

    # Get nearby daughters that are still available
    nearby_daughters = set(nearby_daughters_dict[mother])
    for scored_family in family_stack:
        for daughter in scored_family.family.daughters:
            nearby_daughters.discard(daughter)

    for daughters in itertools.combinations(nearby_daughters, 2):
        scored_family = _score(cache, experiment, graph, mother, daughters[0], daughters[1])
        family_stack_new = family_stack + [scored_family]

        if mother_pos >= len(mothers) - 1:
            yield family_stack_new
        else:
            yield from _scan(cache, experiment, graph, mothers, mother_pos + 1, nearby_daughters_dict, family_stack_new)


def _score(cache: Dict[Family, ScoredFamily], experiment: Experiment, graph: Graph,
           mother: Particle, daughter1: Particle, daughter2: Particle) -> ScoredFamily:
    family = Family(mother, daughter1, daughter2)
    try:
        return cache[family]
    except KeyError:
        scored = ScoredFamily(family, _cell_is_mother_likeliness(experiment, graph, mother, daughter1, daughter2))
        cache[family] = scored
        return scored


def _score_stack(family_stack: List[ScoredFamily]):
    """Calculates the total score of a collection of families."""
    score = 0
    for family in family_stack:
        score += family.score
    return score


def _get_cell_divisions(time_point: TimePoint, graph: Graph) -> Set[Family]:
    families = set()
    for particle in time_point.particles():
        future_preferred_particles = find_preferred_links(graph, particle, find_future_particles(graph, particle))
        if len(future_preferred_particles) >= 2:
            families.add(Family(mother=particle, daughter1=future_preferred_particles.pop(),
                                daughter2=future_preferred_particles.pop()))
    return families


def _find_mothers_and_daughters(time_point: TimePoint, next_time_point: TimePoint,
                               tp_images: ndarray, next_tp_images: ndarray,
                               detection_radius_small: int, detection_radius_large: int) -> SearchResult:
    """Finds likely and less likely mothers and daughters. Mothers are found in the given time point, daughets in the
    next time point.
    """
    search_result = SearchResult()
    for particle in time_point.particles():
        _search_for_mother(particle, tp_images, next_tp_images, search_result,
                           detection_radius_small, detection_radius_large)

    for particle in next_time_point.particles():
        _search_for_daughter(particle, tp_images, next_tp_images, search_result,
                            detection_radius_small, detection_radius_large)
    return search_result


def _search_for_mother(particle: Particle, tp_images: ndarray, next_tp_images: ndarray,
                       search_result: SearchResult, detection_radius_small: int, detection_radius_large: int):
    z = int(particle.z)
    try:
        intensities_previous = normalized_image.get_square(tp_images[z], particle.x, particle.y,
                                                          detection_radius_small)
        intensities_next = normalized_image.get_square(next_tp_images[z], particle.x, particle.y,
                                                      detection_radius_small)
        if numpy.average(intensities_previous) / (numpy.average(intensities_next) + 0.0001) > 2:
            search_result.add_mother(particle)
            return
        intensities_difference = intensities_next - intensities_previous
        if intensities_difference.max() < 0.05 and intensities_difference.min() > -0.05:
            return  # No change in intensity, so just a normal moving cell

        intensities_previous = normalized_image.get_square(tp_images[z], particle.x, particle.y,
                                                          detection_radius_large)
        intensities_next = normalized_image.get_square(next_tp_images[z], particle.x, particle.y,
                                                      detection_radius_large)
        if numpy.average(intensities_previous) / (numpy.average(intensities_next) + 0.0001) > 1.1:
            search_result.add_mother_unsure(particle)
    except IndexError:
        pass  # Cell too close to border of image


def _search_for_daughter(particle: Particle, tp_images: ndarray, next_tp_images: ndarray,
                        search_result: SearchResult, detection_radius_small: int, detection_radius_large: int):
    z = int(particle.z)
    try:
        intensities_next = normalized_image.get_square(next_tp_images[z], particle.x, particle.y,
                                                       detection_radius_small)
        intensities_previous = normalized_image.get_square(tp_images[z], particle.x, particle.y,
                                                           detection_radius_small)
        intensities_difference = intensities_next - intensities_previous
        if numpy.average(intensities_next) / (numpy.average(intensities_previous) + 0.0001) > 2:
            search_result.add_daughter(particle)
            return
        if intensities_difference.max() < 0.05 and intensities_difference.min() > -0.05:
            return  # No change in intensity, so just a normal moving cell

        intensities_next = normalized_image.get_square(next_tp_images[z], particle.x, particle.y,
                                                       detection_radius_large)
        intensities_previous = normalized_image.get_square(tp_images[z], particle.x, particle.y,
                                                           detection_radius_large)
        intensities_difference = intensities_next - intensities_previous
        if intensities_difference.max() > 0.4:
            search_result.add_daughter_unsure(particle)
    except IndexError:
        pass  # Cell too close to border of image


def _cell_is_mother_likeliness(experiment: Experiment, graph: Graph, mother: Particle, daughter1: Particle,
                               daughter2: Particle, mitotic_radius: int = 2):

    mother_image_stack = experiment.get_time_point(mother.time_point_number()).load_images()
    daughter1_image_stack = experiment.get_time_point(daughter1.time_point_number()).load_images()
    mother_image = mother_image_stack[int(mother.z)]
    mother_image_next = daughter1_image_stack[int(mother.z)]
    daughter1_image = daughter1_image_stack[int(daughter1.z)]
    daughter2_image = get_2d_image(experiment, daughter2)
    daughter1_image_prev = mother_image_stack[int(daughter1.z)]
    daughter2_image_prev = mother_image_stack[int(daughter2.z)]

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
    score += score_cell_deaths(graph, mother, daughter1, daughter2)
    score += score_mother_shape(mother, mother_image)
    return score


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


def score_cell_deaths(graph: Graph, mother: Particle, daughter1: Particle, daughter2: Particle):
    score = 0
    for daughter in [daughter1, daughter2]:
        existing_mother = find_preferred_past_particle(graph, daughter)
        if existing_mother == mother:
            continue  # Don't worry, no cell will become dead
        futures = find_preferred_links(graph, existing_mother, find_future_particles(graph, existing_mother))
        if len(futures) >= 2:
            continue  # Don't worry, cell will have another
        score -= 1  # Penalize for creating a dead cell when daughter is removed from the existing mother
    return score


def score_mother_shape(mother: Particle, full_image: ndarray, detection_radius = 16) -> int:
    """Returns a black-and-white image where white is particle and black is background, at least in theory."""

    # Zoom in on mother
    x = int(mother.x)
    y = int(mother.y)
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
