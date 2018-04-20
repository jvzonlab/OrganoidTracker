from typing import Iterable, Set, Optional, Tuple, List

import numpy
from networkx import Graph

import imaging
from imaging import Particle, Experiment, Family, errors, normalized_image
from imaging.normalized_image import ImageEdgeError
from linking import mother_finder
from linking.scoring_system import MotherScoringSystem


def add_mother_scores(experiment: Experiment, graph: Graph, score_system: MotherScoringSystem):
    """Calculates the score for all possible mothers in the graph. (So for all cells with at least two future
    positions.)
    """
    families = mother_finder.find_families(graph, warn_on_many_daughters=False)
    for i in range(len(families)):
        if i % 50 == 0:  # Periodic progress updates
            print("Working on putative family " + str(i) + "/" + str(len(families)))
        family = families[i]
        mother = family.mother
        daughters = list(family.daughters)
        score = score_system.calculate(experiment, mother, daughters[0], daughters[1])
        experiment.get_time_point(mother.time_point_number()).mother_score(family, score)


def __repair_dead_cell(experiment: Experiment, graph: Graph, particle: Particle, used_candidates=set(),
                       remaining_iterations=8) -> bool:
    """Repairs a dead cell, searching for a "fake" mother nearby. Note: if the cell is not repairable, this procedure
    will abort halfway with a ValueError. To prevent this, call this routine with check_only=True first.
    """
    intensity = max(0.1, __get_intensity(experiment, particle))

    future_particles = find_future_particles(graph, particle)
    future_preferred_particles = find_preferred_links(graph, particle, future_particles)

    if len(future_preferred_particles) > 0:
        return False

    # Oops, found dead end. Choose a best match from the future_particles list
    candidates = set(future_particles)
    if len(candidates) > 10:
        return False  # Give up, this is becoming too expensive to calculate
    candidates.difference_update(used_candidates)  # Exclude cells that are already being relinked
    candidates = __remove_relatively_far_away(candidates, particle, tolerance=1.5)
    while True:  # Loop through all possible candidates
        candidate = get_closest_particle_having_a_sister(graph, candidates, particle)
        if candidate is None:  # Ok, ok, choose a less likely one
            candidate = imaging.get_closest_particle(candidates, particle)
        if candidate is None:
            return False  # No more candidates left

        past_of_candidate = find_preferred_past_particle(graph, candidate)
        if past_of_candidate is None:
            return False  # Strange, cell popped up out of nothing. Maybe this is the first time point?

        # Intensity may not increase too much
        candidate_intensity = __get_intensity(experiment, candidate)
        if candidate_intensity / (intensity + 0.0001) > 2:
            candidates.remove(candidate)
            continue

        # Try to change the link
        graph.add_edge(past_of_candidate, candidate, pref=False)
        graph.add_edge(candidate, particle, pref=True)
        remaining_future_of_past_of_candidate = find_preferred_future_particles(graph, past_of_candidate)
        repairable = len(remaining_future_of_past_of_candidate) > 0

        if not repairable and remaining_iterations > 0:
            used_candidates_new = used_candidates.copy()
            used_candidates_new.add(candidate)
            repairable = __repair_dead_cell(experiment, graph, past_of_candidate, used_candidates_new,
                                            remaining_iterations - 1)

        if not repairable:
            # Undo changes
            graph.add_edge(past_of_candidate, candidate, pref=True)
            graph.add_edge(candidate, particle, pref=False)
        else:
            print("Created new link for previously dead " + str(particle) + " towards " + str(
                candidate) + ", replaces link with " + str(past_of_candidate))

            # Show warning if mother with a considerable score has been broken up
            if len(remaining_future_of_past_of_candidate) == 1:
                time_point = experiment.get_time_point(past_of_candidate.time_point_number())
                broken_up_family = Family(past_of_candidate, candidate, *remaining_future_of_past_of_candidate)
                try:
                    score = time_point.mother_score(broken_up_family)
                    if score.is_likely_mother():
                        graph.add_node(past_of_candidate, error=errors.POTENTIALLY_SHOULD_BE_A_MOTHER)
                except KeyError:
                    pass  # Good thing, as this cell doesn't even have a mother score

        if not repairable:
            candidates.remove(candidate)  # This candidate doesn't work out
        else:
            return True  # Worked!


def __get_intensity(experiment: Experiment, particle: Particle, radius: int = 3) -> float:
    try:
        time_point = experiment.get_time_point(particle.time_point_number())
        image = experiment.get_image_stack(time_point)[int(particle.z)]
        return numpy.average(normalized_image.get_square(image, particle.x, particle.y, radius))
    except ImageEdgeError:
        return 0


def __remove_relatively_far_away(particles: Set[Particle], center: Particle, tolerance: float) -> Set[Particle]:
    if len(particles) <= 1:
        return particles
    min_distance_squared = imaging.get_closest_particle(particles, center).distance_squared(center)
    max_distance_squared = min_distance_squared * (tolerance ** 2)
    return {particle for particle in particles if particle.distance_squared(center) <= max_distance_squared}


def fix_no_future_particle(experiment: Experiment, graph: Graph, particle: Particle):
    """This fixes the case where a particle has no future particle lined up"""
    __repair_dead_cell(experiment, graph, particle)


def get_closest_particle_having_a_sister(graph: Graph,
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

        past_of_candidate = find_preferred_past_particle(graph, candidate)
        if past_of_candidate is None:
            return None

        graph.add_edge(past_of_candidate, candidate, pref=False)
        remaining_connections_of_past_of_candidate = find_preferred_links(graph, past_of_candidate, find_future_particles(graph, past_of_candidate))
        graph.add_edge(past_of_candidate, candidate, pref=True)

        if len(remaining_connections_of_past_of_candidate) == 0:
            # Didn't work
            candidates.remove(candidate)
        else:
            return candidate


def downgrade_edges_pointing_to_past(graph: Graph, particle: Particle, allow_deaths: bool = True) -> bool:
    """Removes all edges pointing to the past. When allow_deaths is set to False, the action is cancelled when a
    particle connected to the given particle would become dead (i.e. has no connections to the future left)
    Returns whether all edges were removed, which is always the case if `allow_deaths == True`
    """
    for particle_in_past in find_preferred_links(graph, particle, find_past_particles(graph, particle)):
        remove_error(graph, particle_in_past, errors.POTENTIALLY_NOT_A_MOTHER)  # Indeed not a mother anymore

        graph.add_edge(particle_in_past, particle, pref=False)
        remaining_connections = find_preferred_links(graph, particle_in_past, find_future_particles(graph, particle_in_past))
        if len(remaining_connections) == 0 and not allow_deaths:
            # Oops, that didn't work out. We marked a particle as dead by breaking all its links to the future
            graph.add_edge(particle_in_past, particle, pref=True)
            return False
    return True


def find_preferred_links(graph: Graph, particle: Particle, linked_particles: Iterable[Particle]):
    preferred_particles = set()
    for linked_particle in linked_particles:
        link = graph[particle][linked_particle]
        if "pref" not in link or link["pref"] is True:
            preferred_particles.add(linked_particle)
    return preferred_particles


def find_past_particles(graph: Graph, particle: Particle):
    # all possible connections one step in the past
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.time_point_number() < particle.time_point_number()}


def find_preferred_past_particle(graph: Graph, particle: Particle):
    # the one most likely connection one step in the past
    previous_positions = find_preferred_links(graph, particle, find_past_particles(graph, particle))
    if len(previous_positions) == 0:
        print("Error at " + str(particle) + ": cell popped up out of nothing")
        return None
    if len(previous_positions) > 1:
        print("Error at " + str(particle) + ": cell originated from two different cells")
        return None
    return previous_positions.pop()


def find_future_particles(graph: Graph, particle: Particle) -> Set[Particle]:
    # All possible connections one step in the future
    linked_particles = graph[particle]
    return {linked_particle for linked_particle in linked_particles
            if linked_particle.time_point_number() > particle.time_point_number()}


def find_preferred_future_particles(graph: Graph, particle: Particle) -> Set[Particle]:
    return find_preferred_links(graph, particle, find_future_particles(graph, particle))


def remove_error(graph: Graph, particle: Particle, error: int):
    """Removes the given error associated to the given particle."""
    if "error" in graph.nodes[particle] and graph.nodes[particle]["error"] == error:
        del graph.nodes[particle]["error"]


def with_only_the_preferred_edges(old_graph: Graph):
    graph = Graph()
    for node, data in old_graph.nodes(data=True):
        if not isinstance(node, Particle):
            raise ValueError("Found a node that was not a particle: " + str(node))
        graph.add_node(node, **data)

    for particle_1, particle_2, data in old_graph.edges(data=True):
        if data["pref"]:
            graph.add_edge(particle_1, particle_2)
    return graph


def get_2d_image(experiment: Experiment, particle: Particle):
    images = experiment.get_image_stack(experiment.get_time_point(particle.time_point_number()))
    if images is None:
        raise ValueError("Image for time point " + str(particle.time_point_number()) + " not loaded")
    image = images[int(particle.z)]
    return image