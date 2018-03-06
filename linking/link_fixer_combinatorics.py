import numpy
from networkx import Graph
from numpy import ndarray
import itertools
import math

import imaging
from imaging import Particle, Experiment, TimePoint, normalized_image
from linking.link_fixer import fix_no_future_particle, with_only_the_preferred_edges, find_future_particles, find_preferred_links
from linking_analysis import logical_tests
from typing import Set, Union, Tuple, Dict, List


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


class Family:
    mother: Particle
    daughter1: Particle
    daughter2: Particle
    score: float

    def __init__(self, mother: Particle, daughter1: Particle, daughter2: Particle, score: float):
        self.mother = mother
        self.daughter1 = daughter1
        self.daughter2 = daughter2
        self.score = score

    def __repr__(self):
        return "<m" + ("%.2f" % (self.mother.x + self.mother.y)) \
               + "(d" + ("%.2f" % (self.daughter1.x + self.daughter1.y)) \
               + ", d" + ("%.2f" % (self.daughter2.x + self.daughter2.y)) + ")>"


def prune_links(experiment: Experiment, graph: Graph, detection_radius_small: int, detection_radius_large: int,
                max_distance_mother_daughter: int):
    """Takes a graph with all possible edges between cells, and returns a graph with only the most likely edges.
    mitotic_radius is the radius used to detect whether a cell is undergoing mitosis (i.e. it will have divided itself
    into two in the next time_point). For non-mitotic cells, it must fall entirely within the cell, for mitotic cells it must
    fall partly outside the cell.
    """
    all_mothers = set(); all_daughters = set()

    [fix_no_future_particle(graph, particle) for particle in graph.nodes()]
    for time_point_number in range(experiment.first_time_point_number(), experiment.last_time_point_number() + 1):
        time_point = experiment.get_time_point(time_point_number)
        family_stack = _fix_cell_divisions_for_time_point(experiment, graph, time_point, detection_radius_small,
                                                          detection_radius_large, max_distance_mother_daughter)
        for family in family_stack:
            all_mothers.add(family.mother)
            all_daughters.add(family.daughter1)
            all_daughters.add(family.daughter2)
    [fix_no_future_particle(graph, particle) for particle in graph.nodes()]

    graph = with_only_the_preferred_edges(graph)
    logical_tests.apply(experiment, graph)
    return all_mothers, all_daughters


def _fix_cell_divisions_for_time_point(experiment: Experiment, graph: Graph, time_point: TimePoint,
                        detection_radius_small: int, detection_radius_large: int, max_distance_mother_daughter: int
                        ) -> List[Family]:
    print("Working on time point " + str(time_point.time_point_number()))
    tp_images = time_point.load_images()
    try:
        next_time_point = experiment.get_next_time_point(time_point)
    except KeyError:
        return []  # Last time point, cannot do anything
    next_tp_images = next_time_point.load_images()

    cell_divisions_count = _get_cell_division_count(time_point, graph)
    if cell_divisions_count == 0:
        return []  # No cell divisions here
    possible_mothers_and_daughters = _find_mothers_and_daughters(time_point, next_time_point, tp_images, next_tp_images,
                                                                 detection_radius_small, detection_radius_large)
    mothers, daughters = possible_mothers_and_daughters.find_likely_candidates(max_distance_mother_daughter)
    family_stack = _perform_combinatorics(cell_divisions_count, mothers, daughters,
                                          max_distance=max_distance_mother_daughter)
    print("Mothers: " + str(len(mothers)) + " (expected " + str(cell_divisions_count)
          + ")  Daughters: " + str(len(daughters)) + " Best combination: "
          + str(family_stack))

    return family_stack


def _perform_combinatorics(mothers_pick_amount: int, mothers: Set[Particle], daughters: Set[Particle], max_distance: int
                           ) -> List[Family]:
    # Loop through all possible combinations of N mothers
    nearby_daughters = dict()
    for mother in mothers:
        nearby_daughters[mother] = imaging.get_closest_n_particles(daughters, mother, 4, max_distance=max_distance)

    best_family_stack = []
    best_family_stack_score = 0
    for picked_mothers in itertools.combinations(mothers, mothers_pick_amount):
        for family_stack in _scan(picked_mothers, 0, nearby_daughters):
            score = _score_stack(family_stack)
            if score > best_family_stack_score:
                best_family_stack = family_stack
    return best_family_stack


def _scan(mothers: List[Particle], mother_pos: int, nearby_daughters_dict: Dict[Particle, Set[Particle]],
          family_stack: List[Family] = []):
    mother = mothers[mother_pos]

    # Get nearby daughters that are still available
    nearby_daughters = set(nearby_daughters_dict[mother])
    for family in family_stack:
        nearby_daughters.discard(family.daughter1)
        nearby_daughters.discard(family.daughter2)

    for daughters in itertools.combinations(nearby_daughters, 2):
        score = _score(mother, daughters[0], daughters[1])
        family = Family(mother, daughters[0], daughters[1], score)
        family_stack_new = family_stack + [family]

        if mother_pos >= len(mothers) - 1:
            yield family_stack_new
        else:
            yield from _scan(mothers, mother_pos + 1, nearby_daughters_dict, family_stack_new)


def _score(mother: Particle, daughter1: Particle, daughter2: Particle):
    return 1


def _score_stack(family_stack: List[Family]):
    """Calculates the total score of a collection of families."""
    score = 0
    for family in family_stack:
        score += family.score
    return score


def _get_cell_division_count(time_point: TimePoint, graph: Graph):
    count = 0
    for particle in time_point.particles():
        future_preffered_particles = find_preferred_links(graph, particle, find_future_particles(graph, particle))
        if len(future_preffered_particles) >= 2:
            count += 1
    return count


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
        _search_for_daugher(particle, tp_images, next_tp_images, search_result,
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


def _search_for_daugher(particle: Particle, tp_images: ndarray, next_tp_images: ndarray,
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