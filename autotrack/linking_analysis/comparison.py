from typing import Iterable

import networkx
from networkx import Graph

from autotrack.core import Particle, Experiment
from autotrack.linking import mother_finder
from autotrack.linking_analysis import cell_death_finder


def print_differences(experiment: Experiment):
    _print_links_differences(experiment.particle_links_scratch(), experiment.particle_links())
    _print_mother_differences(experiment.particle_links_scratch(), experiment.particle_links())
    _print_death_differences(experiment)


def _print_mother_differences(automatic_links: Graph, baseline_links: Graph):
    baseline_families = set(mother_finder.find_families(baseline_links))
    automatic_families = set(mother_finder.find_families(automatic_links))

    missed_families = baseline_families.difference(automatic_families)
    made_up_families = automatic_families.difference(baseline_families)
    print("There are " + str(len(missed_families)) + " mother cells that were not recognized")
    _print_cells([family.mother for family in missed_families])

    print("There are " + str(len(made_up_families)) + " mother cells made up by the linking algorithm")
    _print_cells([family.mother for family in made_up_families])


def _print_death_differences(experiment: Experiment):
    baseline_deaths = set(cell_death_finder.find_cell_deaths(experiment, experiment.particle_links()))
    automatic_deaths = set(cell_death_finder.find_cell_deaths(experiment, experiment.particle_links_scratch()))

    missed_deaths = baseline_deaths.difference(automatic_deaths)
    made_up_deaths = automatic_deaths.difference(baseline_deaths)
    print("There are " + str(len(missed_deaths)) + " cell deaths that were not recognized (out of "
          + str(len(baseline_deaths)) + ")")
    _print_cells([family for family in missed_deaths])

    print("There are " + str(len(made_up_deaths)) + " cell deaths made up by the linking algorithm")
    _print_cells([family for family in made_up_deaths])


def _print_cells(cells: Iterable[Particle]):
    lines = 0
    for cell in cells:
        lines += 1
        if lines > 30:
            print("\t...")
            return

        print("\t" + str(cell))


def _print_edges(graph: Graph):
    lines = 0
    for particle1, particle2 in graph.edges():
        lines += 1
        if lines > 30:
            print("\t...")
            return

        print("\t" + str(particle1) + "---" + str(particle2))


def _print_links_differences(automatic_links: Graph, baseline_links: Graph):
    print("There are " + str(baseline_links.number_of_nodes()) + " cells and "+ str(baseline_links.number_of_edges())
          + " links in the baseline results.")
    missed_links = networkx.difference(baseline_links, automatic_links);
    made_up_links = networkx.difference(automatic_links, baseline_links);

    print("There are " + str(networkx.number_of_edges(missed_links)) + " connections missed in the automatic results:")
    _print_edges(missed_links)

    print("There are " + str(networkx.number_of_edges(made_up_links)) + " connections made by the automatic linker"
          " that did not exist in the manual results:")
    _print_edges(made_up_links)