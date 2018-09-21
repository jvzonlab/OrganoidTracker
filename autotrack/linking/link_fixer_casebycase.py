# The following rules can be used:
# * Every cell must have one or two cells in the next image
# * Every cell must have exactly one cell in the previous image

import itertools
from typing import Set, Optional, Tuple

from networkx import Graph

from autotrack.core import Particle, Experiment, Family
from autotrack.linking import logical_tests, errors
from autotrack.linking.link_fixer import downgrade_edges_pointing_to_past, find_preferred_links, find_preferred_past_particle, \
    find_future_particles, remove_error, with_only_the_preferred_edges, add_mother_scores, fix_no_future_particle, \
    get_closest_particle_having_a_sister, find_preferred_future_particles
from autotrack.linking import MotherScoringSystem


def prune_links(experiment: Experiment, graph: Graph, score_system: MotherScoringSystem) -> Graph:
    """Takes a graph with all possible edges between cells, and returns a graph with only the most likely edges./
    """
    add_mother_scores(experiment, graph, score_system)
    for i in range(2):
        [fix_no_future_particle(experiment, graph, particle) for particle in graph.nodes()]
        [_fix_cell_division_mother(experiment, graph, particle)
            for particle in graph.nodes()]
        [_fix_cell_division_daughters(experiment, graph, particle)
            for particle in graph.nodes()]

    graph = with_only_the_preferred_edges(graph)
    logical_tests.apply(experiment, graph)
    return graph


def _fix_cell_division_daughters(experiment: Experiment, graph: Graph, particle: Particle):
    time_point = experiment.get_time_point(particle.time_point_number())
    future_particles = find_future_particles(graph, particle)
    future_preferred_particles = find_preferred_links(graph, particle, future_particles)

    if len(future_preferred_particles) != 2 or len(future_particles) <= 2:
        return

    current_daughter1 = future_preferred_particles.pop()
    current_daughter2 = future_preferred_particles.pop()
    current_family = Family(particle, current_daughter1, current_daughter2)
    try:
        current_score = time_point.mother_score(current_family).total()
    except KeyError:
        print("Cannot observe daughters of " + str(particle) + " - no score specified")
        return

    best_daughter1 = None
    best_daughter2 = None
    best_score = current_score

    for daughter1, daughter2 in itertools.combinations(future_particles, 2):
        intersection = {daughter1, daughter2} & current_family.daughters
        if len(intersection) != 1:
            continue  # Exactly one daughter must be new
        if _has_sister(graph, daughter1) and _has_sister(graph, daughter2):
            # Two nearby cell divisions, don't try to recombine
            # (maybe this could be interesting for the future)
            continue

        # Check if this combination is better
        new_family = Family(particle, daughter1, daughter2)
        score = time_point.mother_score(new_family).total()
        if score > best_score:
            best_daughter1 = daughter1
            best_daughter2 = daughter2
            best_score = score

    if best_score / current_score >= 4/3:  # Improvement is possible
        # Find the cells
        intersection = {best_daughter1, best_daughter2} & current_family.daughters

        removed_daughter = current_family.daughters.difference(intersection).pop()
        remaining_daughter = intersection.pop()
        new_daughter = best_daughter1 if best_daughter1 != remaining_daughter else best_daughter2
        old_parent_of_new_daughter = find_preferred_past_particle(graph, new_daughter)

        # Switches the daughter (and binds the old parent to the newly removed daughter)
        graph.add_edge(new_daughter, old_parent_of_new_daughter, pref=False)
        graph.add_edge(particle, new_daughter, pref=True)
        graph.add_edge(particle, removed_daughter, pref=False)
        graph.add_edge(removed_daughter, old_parent_of_new_daughter, pref=True)

        print("Set " + str(new_daughter) + " (score " + str(best_score) + ") as the new daughter of " + str(particle) +
              ", instead of " + str(removed_daughter) + " (score " + str(current_score) + ")")
    if best_score > current_score:
        graph.add_node(particle, error=errors.POTENTIALLY_WRONG_DAUGHTERS)


def _fix_cell_division_mother(experiment: Experiment, graph: Graph, particle: Particle):
    """Checks if there isn't a mother nearby that is a worse mother than this cell would be. If yes, one daughter over
    there is removed and placed under this cell."""
    future_particles = find_future_particles(graph, particle)
    future_preferred_particles = find_preferred_links(graph, particle, future_particles)

    if len(future_particles) < 2 or len(future_preferred_particles) == 0:
        return  # Surely not a mother cell

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
    if len(children_of_current_mother_of_daughter2) > 2:
        children_of_current_mother_of_daughter2 = children_of_current_mother_of_daughter2[0:2]

    family = Family(particle, *two_daughters)
    current_family = Family(current_mother_of_daughter2, *children_of_current_mother_of_daughter2)
    try:
        score = experiment.get_time_point(particle.time_point_number()).mother_score(family).total()
        current_parent_score = experiment.get_time_point(current_mother_of_daughter2.time_point_number())\
            .mother_score(current_family).total()
    except KeyError:
        print("Cannot compare " + str(particle) + " - no mother score set")
        return
    # Printing of warnings
    if abs(score - current_parent_score) <= 2:
        # Not sure
        if score > current_parent_score:
            graph.add_node(particle, error=errors.POTENTIALLY_NOT_A_MOTHER)
        else:
            graph.add_node(current_mother_of_daughter2, error=errors.POTENTIALLY_NOT_A_MOTHER)
    else:  # Remove any existing errors, they will be outdated
        remove_error(graph, particle, errors.POTENTIALLY_NOT_A_MOTHER)
        remove_error(graph, current_mother_of_daughter2, errors.POTENTIALLY_NOT_A_MOTHER)

    # Parent replacement
    if score > current_parent_score:
        # Replace parent
        print("Let " + str(particle) + " (score " + str(score) + ") replace " + str(
            current_mother_of_daughter2) + " (score " + str(current_parent_score) + ")")
        downgrade_edges_pointing_to_past(graph, daughter2)  # Removes old mother
        graph.add_edge(particle, daughter2, pref=True)
    else:
        print("Didn't let " + str(particle) + " (score " + str(score) + ") replace " + str(
            current_mother_of_daughter2) + " (score " + str(current_parent_score) + ")")


#
# Helper functions below
#


def _has_sister(graph: Graph, particle: Particle) -> bool:
    mother = find_preferred_past_particle(graph, particle)
    if mother is None:
        return False
    return len(find_preferred_future_particles(graph, mother)) > 2


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




