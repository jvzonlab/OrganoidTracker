import networkx
from networkx import Graph
from linking_analysis import mother_finder
from typing import Iterable
from imaging import Particle

def print_differences(automatic_links: Graph, baseline_links: Graph):
    _print_links_differences(automatic_links, baseline_links)
    _print_mother_differences(automatic_links, baseline_links)


def _print_mother_differences(automatic_links: Graph, baseline_links: Graph):
    baseline_mothers = mother_finder.find_mothers(baseline_links)
    automatic_mothers = mother_finder.find_mothers(automatic_links)

    missed_mothers = baseline_mothers.difference(automatic_mothers)
    made_up_mothers = automatic_mothers.difference(baseline_mothers)
    print("There are " + str(len(missed_mothers)) + " mother cells that were not recognized")
    _print_cells(missed_mothers)

    print ("There are " + str(len(made_up_mothers)) + " mother cells made up by the linking algorithm")
    _print_cells(made_up_mothers)


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
    print("There are " + str(baseline_links.number_of_edges()) + " connections in the baseline results.")
    missed_links = networkx.difference(baseline_links, automatic_links);
    made_up_links = networkx.difference(automatic_links, baseline_links);

    print("There are " + str(networkx.number_of_edges(missed_links)) + " connections missed in the automatic results:")
    _print_edges(missed_links)

    print("There are " + str(networkx.number_of_edges(made_up_links)) + " connections made by the automatic linker"
          " that did not exist in the manual results:")
    _print_edges(made_up_links)