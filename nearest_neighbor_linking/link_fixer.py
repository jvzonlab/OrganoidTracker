# The following rules can be used:
# * Every cell must have one or two cells in the next image
# * Every cell must have exactly one cell in the previous image

import imaging
import numpy
from networkx import Graph
from typing import Iterable, Set, Optional, Tuple
from imaging import Particle, Experiment, cell, errors
from imaging.image_helper import Image2d


def prune_links(experiment: Experiment, graph: Graph, mitotic_radius: int) -> Graph:
    """Takes a graph with all possible edges between cells, and returns a graph with only the most likely edges.
    mitotic_radius is the radius used to detect whether a cell is undergoing mitosis (i.e. it will have divided itself
    into two in the next frame). For non-mitotic cells, it must fall entirely within the cell, for mitotic cells it must
    fall partly outside the cell.
    """
    last_frame_number = experiment.last_frame_number()

    [_fix_no_future_particle(graph, particle, last_frame_number) for particle in graph.nodes()]
    [_fix_cell_divisions(experiment, graph, particle, mitotic_radius) for particle in graph.nodes()]

    return _with_only_the_preferred_edges(graph)


def _fix_no_future_particle(graph: Graph, particle: Particle, last_frame_number: int):
    """This fixes the case where a particle has no future particle lined up"""
    future_particles = _find_future_particles(graph, particle)
    future_preferred_particles = _find_preferred_links(graph, particle, future_particles)

    if len(future_preferred_particles) > 0:
        return

    # Oops, found dead end. Choose a best match from the future_particles list
    newly_matched_future_particle = imaging.get_closest_particle(future_particles, particle)
    if newly_matched_future_particle is None:
        if particle.frame_number() != last_frame_number:
            graph.add_node(particle, error=errors.NO_FUTURE_POSITION)
        return

    # Replace edge
    _downgrade_edges_pointing_to_past(graph, newly_matched_future_particle)
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

    two_daughters = _get_two_daughters(particle, future_preferred_particles, future_particles)
    if two_daughters is None:
        raise ValueError("Unable to find two daughters, even though there were at least two future_particles")
    score = _cell_is_mother_likeliness(experiment, graph, particle, two_daughters[0], two_daughters[1], mitotic_radius)

    # Daughter1 surely is in preferred_particles, but maybe daughter2 not yet. If so, we might need to declare this cell
    # as a mother, and "undeclare" another cell from being one
    daughter2 = two_daughters[1]
    if daughter2 in future_preferred_particles:
        return  # Nothing to fix
    current_mother_of_daughter2 = _find_past_particle(graph, daughter2)
    children_of_current_parent_of_daughter2 = list(_find_preferred_links(graph, current_mother_of_daughter2,
                                                   _find_future_particles(graph, current_mother_of_daughter2)))
    if len(children_of_current_parent_of_daughter2) < 2:
        return # Cannot decouple current parent from daughter2, as then the current parent would be a dead cell

    current_parent_score = _cell_is_mother_likeliness(experiment, graph, current_mother_of_daughter2,
                                                      children_of_current_parent_of_daughter2[0],
                                                      children_of_current_parent_of_daughter2[1], mitotic_radius)
    if score > current_parent_score:
        # Replace parent
        _downgrade_edges_pointing_to_past(graph, daughter2) # Removes old mother
        graph.add_edge(particle, daughter2, pref=True)

    if abs(score - current_parent_score) < 3:
        # Not sure
        if score > current_parent_score:
            graph.add_node(particle, error=errors.POTENTIALLY_NOT_A_MOTHER)
        else:
            graph.add_node(particle, error=errors.POTENTIALLY_SHOULD_BE_A_MOTHER)

#
# Helper functions below
#


def _downgrade_edges_pointing_to_past(graph: Graph, particle: Particle, allow_deaths: bool = True) -> bool:
    """Removes all edges pointing to the past. When allow_deaths is set to False, the action is cancelled when a
    particle connected to the given particle would become dead (i.e. has no connections to the future left)
    Returns whether all edges were removed, which is always the case if `allow_deaths == True`
    """
    for particle_in_past in _find_past_particles(graph, particle):
        graph.add_edge(particle_in_past, particle, pref=False)
        remaining_connections = _find_preferred_links(graph, particle_in_past, _find_future_particles(graph, particle_in_past))
        if len(remaining_connections) == 0 and not allow_deaths:
            # Oops, that didn't work out. We marked a particle as dead by breaking all its links to the future
            graph.add_edge(particle_in_past, particle, pref=True)
            return False
    return True


def _find_preferred_links(graph: Graph, particle: Particle, linked_particles: Iterable[Particle]):
    return {linked_particle for linked_particle in linked_particles
            if graph[particle][linked_particle]["pref"] == True}


def _find_past_particles(graph: Graph, particle: Particle):
    # all possible connections one step in the past
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.frame_number() < particle.frame_number()}


def _find_past_particle(graph: Graph, particle: Particle):
    # the one most likely connection one step in the past
    previous_positions = _find_preferred_links(graph, particle, _find_past_particles(graph, particle))
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
            if linked_particle.frame_number() > particle.frame_number()}


def _with_only_the_preferred_edges(old_graph: Graph):
    graph = Graph()
    for node, data in old_graph.nodes(data=True):
        if not isinstance(node, Particle):
            raise ValueError("Found a node that was not a particle: " + str(node))
        if "error" in data:
            error = data["error"]
            print(errors.get_severity(error).name + " at " + str(node) + ": " + errors.get_message(error))
        graph.add_node(node, **data)

    for particle_1, particle_2, data in old_graph.edges(data=True):
        if data["pref"]:
            graph.add_edge(particle_1, particle_2)
    return graph


def _get_2d_image(experiment: Experiment, particle: Particle):
    images = experiment.get_frame(particle.frame_number()).load_images()
    if images is None:
        raise ValueError("Image for frame " + str(particle.frame_number()) + " not loaded")
    image = images[int(particle.z)]
    return Image2d(image)


def _get_two_daughters(mother: Particle, preferred_particles: Set[Particle], all_particles: Set[Particle]) \
        -> Optional[Tuple[Particle, Particle]]:
    """Gets a list with two daughter cells at positions 0 and 1. First, particles from te preferred lists are chosen,
    then the nearest from the other list are chosen.
    """
    result = set(preferred_particles)
    in_consideration = set(all_particles)
    in_consideration = in_consideration.difference(result)

    while len(result) < 2:
        nearest = imaging.get_closest_particle(in_consideration, mother)
        if nearest is None:
            return None # Simply not enough cells provided
        result.add(nearest)
        in_consideration.remove(nearest)

    return result.pop(), result.pop()


def _get_angle(a: Particle, b: Particle, c: Particle):
    """Gets the angle ∠ABC"""
    ba = numpy.array([a.x - b.x, a.y - b.y])
    bc = numpy.array([c.x - b.x, c.y - b.y])

    cosine_angle = numpy.dot(ba, bc) / (numpy.linalg.norm(ba) * numpy.linalg.norm(bc))
    angle = numpy.arccos(cosine_angle)

    return numpy.degrees(angle)


_cached_intensities = None


def _get_average_intensity_at(experiment: Experiment, particle: Particle):
    return _get_2d_image(experiment, particle).get_average_intensity_at(int(particle.x), int(particle.y))


def _cell_is_mother_likeliness(experiment: Experiment, graph: Graph, mother: Particle, daughter1: Particle,
                               daughter2: Particle, mitotic_radius: int = 2, min_cell_age: int = 4):
    global _cached_intensities

    image = _get_2d_image(experiment, mother)
    _cached_intensities = image.get_intensities_square(int(mother.x), int(mother.y),
                                                       mitotic_radius, _cached_intensities)

    score = 0

    # Mother cell must have high intensity and contrast
    min_value = numpy.amin(_cached_intensities)
    max_value = numpy.amax(_cached_intensities)
    score += max_value * 2 # The higher intensity, the better: the DNA is concentrated
    score += (max_value - min_value) # High contrast is also desirable, as there are parts where there is no DNA

    # Daughter cells must move to opposite angles
    angle = _get_angle(daughter1, mother, daughter2)
    if angle < 80 or angle > 280:
        score -= 4  # Very bad, daughters usually go in opposite directions
    elif angle < 120 or angle > 240:
        score -= 1  # Less bad

    # Daughter cells must keep some distance from mother
    distance1_squared = abs(mother.x - daughter1.x) ** 2 + abs(mother.y - daughter1.y) ** 2
    if distance1_squared < 7 ** 2:
        score -= 1
    distance2_squared = abs(mother.x - daughter2.x) ** 2 + abs(mother.y - daughter2.y) ** 2
    if distance2_squared < 7 ** 2:
        score -= 1

    # Daughter cells must have almost the same intensity
    daughter1_intensity = _get_average_intensity_at(experiment, daughter1)
    daughter2_intensity = _get_average_intensity_at(experiment, daughter2)
    intensity_difference = abs(daughter1_intensity - daughter2_intensity)
    score -= intensity_difference * 2

    # Mothers cannot be too young
    age = cell.get_age(experiment, graph, mother)
    if age is not None and age < min_cell_age:
        score -= 5 # Severe punishment, as this is biologically impossible

    return score