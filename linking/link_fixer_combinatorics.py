import itertools
from typing import Set, Dict, List, Iterable, Tuple

import numpy
from networkx import Graph
from numpy import ndarray

import imaging
from imaging import Particle, Experiment, TimePoint, ScoredFamily, normalized_image
from imaging.normalized_image import ImageEdgeError
from linking import Parameters, logical_tests
from linking.link_fixer import fix_no_future_particle, with_only_the_preferred_edges, find_future_particles, \
    find_preferred_links, downgrade_edges_pointing_to_past
from linking.scoring_system import MotherScoringSystem
from linking.rational_scoring_system import RationalScoringSystem
from linking.mother_finder import Family


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


def prune_links(experiment: Experiment, graph: Graph, parameters: Parameters) -> Tuple[Graph, Set[Family]]:
    """Takes a graph with all possible edges between cells, and returns a graph with only the most likely edges.
    mitotic_radius is the radius used to detect whether a cell is undergoing mitosis (i.e. it will have divided itself
    into two in the next time_point). For non-mitotic cells, it must fall entirely within the cell, for mitotic cells it must
    fall partly outside the cell.
    """
    all_families = set()

    [fix_no_future_particle(experiment, graph, particle) for particle in graph.nodes()]
    for time_point_number in range(experiment.first_time_point_number(), experiment.last_time_point_number() + 1):
        time_point = experiment.get_time_point(time_point_number)
        scored_families = _fix_cell_divisions_for_time_point(experiment, graph, time_point,  parameters)
        for scored_family in scored_families:
            all_families.add(scored_family.family)

    #[fix_no_future_particle(graph, particle) for particle in graph.nodes()]

    graph = with_only_the_preferred_edges(graph)
    logical_tests.apply(experiment, graph)
    return graph, all_families


def _fix_cell_divisions_for_time_point(experiment: Experiment, graph: Graph, time_point: TimePoint,
                                       parameters: Parameters) -> List[ScoredFamily]:
    print("Working on time point " + str(time_point.time_point_number()))

    # Load images
    tp_images = experiment.get_image_stack(time_point)
    try:
        next_time_point = experiment.get_next_time_point(time_point)
    except KeyError:
        return []  # Last time point, cannot do anything
    next_tp_images = experiment.get_image_stack(next_time_point)

    # Calculate the number of expected cell divisions based on cell count
    expected_cell_divisions = len(next_time_point.particles()) - len(time_point.particles())
    if expected_cell_divisions <= 0:
        return []  # No cell divisions here

    # Find potential mothers and daughters, find best combinations
    possible_mothers_and_daughters = _find_mothers_and_daughters(time_point, next_time_point, tp_images, next_tp_images,
                                                                 parameters.intensity_detection_radius,
                                                                 parameters.intensity_detection_radius_large)
    mothers, daughters = possible_mothers_and_daughters.find_likely_candidates(parameters.max_distance)
    family_stack = _perform_combinatorics(experiment, graph, expected_cell_divisions, mothers, daughters, parameters)
    print("Mothers: " + str(len(mothers)) + " (expected " + str(expected_cell_divisions)+ ")  Daughters: "
          + str(len(daughters)) + " Best combination: " + str(family_stack))

    # Apply
    existing_cell_divisions = _get_existing_cell_divisions(time_point, graph)
    _replace_divisions(graph, old_divisions=existing_cell_divisions, new_divisions=family_stack)

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
                           daughters: Set[Particle], parameters: Parameters) -> List[ScoredFamily]:
    # Loop through all possible combinations of N mothers, and picks the best combination of mothers and daughters
    nearby_daughters = dict()
    for mother in mothers:
        nearby_daughters[mother] = imaging.get_closest_n_particles(daughters, mother, 4, max_distance=parameters.max_distance)

    scoring_system = RationalScoringSystem(parameters.intensity_detection_radius, parameters.shape_detection_radius)
    best_family_stack = []
    best_family_stack_score = 0
    for picked_mothers in itertools.combinations(mothers, mothers_pick_amount):
        for family_stack in _scan(experiment, graph, scoring_system, picked_mothers, 0, nearby_daughters, dict()):
            score = _score_stack(family_stack)
            if score > best_family_stack_score:
                best_family_stack = family_stack
                best_family_stack_score = score
    return best_family_stack


def _scan(experiment: Experiment, graph: Graph, score_system: MotherScoringSystem, mothers: List[Particle],
          mother_pos: int, nearby_daughters_dict: Dict[Particle, Set[Particle]],
          already_scored_dict: Dict[Family, ScoredFamily], family_stack: List[ScoredFamily] = []):
    mother = mothers[mother_pos]

    # Get nearby daughters that are still available
    nearby_daughters = set(nearby_daughters_dict[mother])
    for scored_family in family_stack:
        for daughter in scored_family.family.daughters:
            nearby_daughters.discard(daughter)

    for daughters in itertools.combinations(nearby_daughters, 2):
        scored_family = _score(already_scored_dict, experiment, graph, score_system, mother, daughters[0], daughters[1])
        family_stack_new = family_stack + [scored_family]

        if mother_pos >= len(mothers) - 1:
            yield family_stack_new
        else:
            yield from _scan(experiment, graph, score_system, mothers, mother_pos + 1, nearby_daughters_dict,
                             already_scored_dict, family_stack_new)


def _score(cache: Dict[Family, ScoredFamily], experiment: Experiment, graph: Graph, scoring: MotherScoringSystem,
           mother: Particle, daughter1: Particle, daughter2: Particle) -> ScoredFamily:
    family = Family(mother, daughter1, daughter2)
    if family in cache:
        return cache[family]
    else:
        scored = ScoredFamily(family, scoring.calculate(experiment, mother, daughter1, daughter2))
        cache[family] = scored
        return scored


def _score_stack(family_stack: List[ScoredFamily]):
    """Calculates the total score of a collection of families."""
    score = 0
    for family in family_stack:
        score += family.score.total()
    return score


def _get_existing_cell_divisions(time_point: TimePoint, graph: Graph) -> Set[Family]:
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
    except ImageEdgeError:
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
    except ImageEdgeError:
        pass  # Cell too close to border of image
