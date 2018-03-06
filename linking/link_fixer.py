from typing import Iterable, Set, Optional, Tuple, List

from networkx import Graph

import imaging
from imaging import Particle, Experiment, cell, errors, normalized_image


def fix_no_future_particle(graph: Graph, particle: Particle):
    """This fixes the case where a particle has no future particle lined up"""
    future_particles = find_future_particles(graph, particle)
    future_preferred_particles = find_preferred_links(graph, particle, future_particles)

    if len(future_preferred_particles) > 0:
        return

    # Oops, found dead end. Choose a best match from the future_particles list
    newly_matched_future_particle = get_closest_particle_having_a_sister(graph, future_particles, particle)
    if newly_matched_future_particle is None:
        return False

    # Replace edge
    downgrade_edges_pointing_to_past(graph, newly_matched_future_particle)
    print("Created new link for previously dead " + str(particle) + " towards " + str(newly_matched_future_particle))
    graph.add_edge(particle, newly_matched_future_particle, pref=True)


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
    remove_error(graph, particle)  # Remove any errors, they will not be up to date anymore
    for particle_in_past in find_preferred_links(graph, particle, find_past_particles(graph, particle)):
        graph.add_edge(particle_in_past, particle, pref=False)
        remaining_connections = find_preferred_links(graph, particle_in_past, find_future_particles(graph, particle_in_past))
        if len(remaining_connections) == 0 and not allow_deaths:
            # Oops, that didn't work out. We marked a particle as dead by breaking all its links to the future
            graph.add_edge(particle_in_past, particle, pref=True)
            return False
    return True


def find_preferred_links(graph: Graph, particle: Particle, linked_particles: Iterable[Particle]):
    return {linked_particle for linked_particle in linked_particles
            if graph[particle][linked_particle]["pref"] is True}


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


def remove_error(graph: Graph, particle: Particle):
    """Removes any error message associated to the given particle"""
    if "error" in graph.nodes[particle]:
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
    images = experiment.get_time_point(particle.time_point_number()).load_images()
    if images is None:
        raise ValueError("Image for time point " + str(particle.time_point_number()) + " not loaded")
    image = images[int(particle.z)]
    return image