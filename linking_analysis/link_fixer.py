# The following rules can be used:
# * Every cell must have one or two cells in the next image
# * Every cell must have exactly one cell in the previous image

from typing import Iterable, Set, Optional, Tuple, List

import numpy
from networkx import Graph
from numpy import ndarray

import imaging
from imaging import Particle, Experiment, cell, errors, normalized_image
from linking_analysis import logical_tests


def prune_links(experiment: Experiment, graph: Graph, mitotic_radius: int) -> Graph:
    """Takes a graph with all possible edges between cells, and returns a graph with only the most likely edges.
    mitotic_radius is the radius used to detect whether a cell is undergoing mitosis (i.e. it will have divided itself
    into two in the next time_point). For non-mitotic cells, it must fall entirely within the cell, for mitotic cells it must
    fall partly outside the cell.
    """
    for i in range(2):
        [_fix_no_future_particle(graph, particle) for particle in graph.nodes()]
        [_fix_cell_divisions(experiment, graph, particle, mitotic_radius) for particle in graph.nodes()]

    graph = _with_only_the_preferred_edges(graph)
    logical_tests.apply(experiment, graph)
    return graph


def _fix_no_future_particle(graph: Graph, particle: Particle):
    """This fixes the case where a particle has no future particle lined up"""
    future_particles = _find_future_particles(graph, particle)
    future_preferred_particles = _find_preferred_links(graph, particle, future_particles)

    if len(future_preferred_particles) > 0:
        return

    # Oops, found dead end. Choose a best match from the future_particles list
    newly_matched_future_particle = _get_closest_particle_having_a_sister(graph, future_particles, particle)
    if newly_matched_future_particle is None:
        return False

    # Replace edge
    _downgrade_edges_pointing_to_past(graph, newly_matched_future_particle)
    print("Created new link for previously dead " + str(particle) + " towards " + str(newly_matched_future_particle))
    graph.add_edge(particle, newly_matched_future_particle, pref=True)


def _fix_cell_divisions(experiment: Experiment, graph: Graph, particle: Particle, mitotic_radius: int):
    global _cached_intensities

    future_particles = _find_future_particles(graph, particle)
    future_preferred_particles = _find_preferred_links(graph, particle, future_particles)

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
    current_mother_of_daughter2 = _find_preferred_past_particle(graph, daughter2)
    children_of_current_mother_of_daughter2 = list(_find_preferred_links(graph, current_mother_of_daughter2,
                                                   _find_future_particles(graph, current_mother_of_daughter2)))
    if len(children_of_current_mother_of_daughter2) < 2:
        # The _get_two_daughters should have checked for this
        raise ValueError("No nearby mother available for " + str(particle))

    score = _cell_is_mother_likeliness(experiment, graph, particle, two_daughters[0], two_daughters[1], mitotic_radius)
    current_parent_score = _cell_is_mother_likeliness(experiment, graph, current_mother_of_daughter2,
                                                      children_of_current_mother_of_daughter2[0],
                                                      children_of_current_mother_of_daughter2[1], mitotic_radius)
    if abs(score - current_parent_score) <= 1:
        # Not sure
        if score > current_parent_score:
            graph.add_node(particle, error=errors.POTENTIALLY_NOT_A_MOTHER)
            graph.add_node(current_mother_of_daughter2, error=errors.POTENTIALLY_SHOULD_BE_A_MOTHER)
        else:
            graph.add_node(particle, error=errors.POTENTIALLY_SHOULD_BE_A_MOTHER)
            graph.add_node(current_mother_of_daughter2, error=errors.POTENTIALLY_NOT_A_MOTHER)
    else:  # Remove any existing errors, they will be outdated
        _remove_error(graph, particle)
        _remove_error(graph, current_mother_of_daughter2)

    if score > current_parent_score:
        # Replace parent
        print("Let " + str(particle) + " (score " + str(score) + ") replace " + str(
            current_mother_of_daughter2) + " (score " + str(current_parent_score) + ")")
        _downgrade_edges_pointing_to_past(graph, daughter2)  # Removes old mother
        graph.add_edge(particle, daughter2, pref=True)
        return True
    else:
        print("Didn't let " + str(particle) + " (score " + str(score) + ") replace " + str(
            current_mother_of_daughter2) + " (score " + str(current_parent_score) + ")")
        return False


#
# Helper functions below
#


def _get_closest_particle_having_a_sister(graph: Graph,
                                          candidates_list: Iterable[Particle], center: Particle) -> Optional[Particle]:
    """This function gets the closest particle relative to a given center that has a sister. That stipulation may seem
    strange on the first sight, but it is very useful. A lot of cells with two future positions are authomatically
    recognized as a mother with two daughters. However, this may have been a mistake. Maybe the cell has only one
    future position, and the other "daughter" actually belongs to (is a future position of) another cell.
    """
    candidates = set(candidates_list)
    while True:
        candidate = imaging.get_closest_particle(candidates, center)
        if candidate is None:
            return None  # No more candidates left

        past_of_candidate = _find_preferred_past_particle(graph, candidate)

        graph.add_edge(past_of_candidate, candidate, pref=False)
        remaining_connections_of_past_of_candidate = _find_preferred_links(graph, past_of_candidate, _find_future_particles(graph, past_of_candidate))
        graph.add_edge(past_of_candidate, candidate, pref=True)

        if len(remaining_connections_of_past_of_candidate) == 0:
            # Didn't work
            candidates.remove(candidate)
        else:
            return candidate


def _downgrade_edges_pointing_to_past(graph: Graph, particle: Particle, allow_deaths: bool = True) -> bool:
    """Removes all edges pointing to the past. When allow_deaths is set to False, the action is cancelled when a
    particle connected to the given particle would become dead (i.e. has no connections to the future left)
    Returns whether all edges were removed, which is always the case if `allow_deaths == True`
    """
    _remove_error(graph, particle)  # Remove any errors, they will not be up to date anymore
    for particle_in_past in _find_preferred_links(graph, particle, _find_preferred_past_particles(graph, particle)):
        graph.add_edge(particle_in_past, particle, pref=False)
        remaining_connections = _find_preferred_links(graph, particle_in_past, _find_future_particles(graph, particle_in_past))
        if len(remaining_connections) == 0 and not allow_deaths:
            # Oops, that didn't work out. We marked a particle as dead by breaking all its links to the future
            graph.add_edge(particle_in_past, particle, pref=True)
            return False
    return True


def _find_preferred_links(graph: Graph, particle: Particle, linked_particles: Iterable[Particle]):
    return {linked_particle for linked_particle in linked_particles
            if graph[particle][linked_particle]["pref"] is True}


def _find_preferred_past_particles(graph: Graph, particle: Particle):
    # all possible connections one step in the past
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.time_point_number() < particle.time_point_number()}


def _find_preferred_past_particle(graph: Graph, particle: Particle):
    # the one most likely connection one step in the past
    previous_positions = _find_preferred_links(graph, particle, _find_preferred_past_particles(graph, particle))
    if len(previous_positions) == 0:
        print("Error at " + str(particle) + ": cell popped up out of nothing")
        return None
    if len(previous_positions) > 1:
        print("Error at " + str(particle) + ": cell originated from two different cells")
        return None
    return previous_positions.pop()


def _find_future_particles(graph: Graph, particle: Particle):
    # All possible connections one step in the future
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.time_point_number() > particle.time_point_number()}


def _remove_error(graph: Graph, particle: Particle):
    """Removes any error message associated to the given particle"""
    if "error" in graph.nodes[particle]:
        del graph.nodes[particle]["error"]


def _with_only_the_preferred_edges(old_graph: Graph):
    graph = Graph()
    for node, data in old_graph.nodes(data=True):
        if not isinstance(node, Particle):
            raise ValueError("Found a node that was not a particle: " + str(node))
        graph.add_node(node, **data)

    for particle_1, particle_2, data in old_graph.edges(data=True):
        if data["pref"]:
            graph.add_edge(particle_1, particle_2)
    return graph


def _get_2d_image(experiment: Experiment, particle: Particle):
    images = experiment.get_time_point(particle.time_point_number()).load_images()
    if images is None:
        raise ValueError("Image for time point " + str(particle.time_point_number()) + " not loaded")
    image = images[int(particle.z)]
    return image


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
        nearest = _get_closest_particle_having_a_sister(graph, in_consideration, mother)
        if nearest is None:
            return None # Simply not enough cells provided
        result.append(nearest)
        in_consideration.remove(nearest)

    return (result[0], result[1])


def _cell_is_mother_likeliness(experiment: Experiment, graph: Graph, mother: Particle, daughter1: Particle,
                               daughter2: Particle, mitotic_radius: int = 2, min_cell_age: int = 2):

    mother_image_stack = experiment.get_time_point(mother.time_point_number()).load_images()
    daughter1_image_stack = experiment.get_time_point(daughter1.time_point_number()).load_images()
    mother_image = mother_image_stack[int(mother.z)]
    mother_image_next = daughter1_image_stack[int(mother.z)]
    daughter1_image = daughter1_image_stack[int(daughter1.z)]
    daughter2_image = _get_2d_image(experiment, daughter2)
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
