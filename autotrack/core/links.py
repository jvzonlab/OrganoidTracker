from enum import Enum
from typing import Optional

import networkx
from networkx import Graph

from autotrack.core.particles import Particle


class ParticleLinks:
    """Represents all particle links. Two different networks can be specified (called baseline and scratch), so that
    comparisons become possible. Care has been taken to ensure that the node sets of both linking networks are
    equal, so that comparisons between the networks are easier."""

    __graph: Optional[Graph] = None

    def add_links(self, graph: Graph):
        """Adds all links from the graph. Existing link are not removed."""
        if self.__graph is None:
            self.__graph = graph
        else:
            self.__graph.add_nodes_from(graph.nodes)
            self.__graph.add_edges_from(graph.edges)

    def remove_all_links(self):
        """Removes all links in the experiment."""
        self.__graph = None

    @property
    def graph(self) -> Optional[Graph]:
        """The linking data. May be None."""
        return self.__graph

    def remove_links_of_particle(self, particle: Particle):
        """Removes all links from and to the particle."""
        if self.__graph is not None and particle in self.__graph:
            self.__graph.remove_node(particle)

    def replace_particle(self, old_position: Particle, position_new: Particle):
        """Replaces one particle with another. The old particle is removed from the graph, the new one is added. All
        links will be moved over to the new particle"""
        mapping = {old_position: position_new}
        if self.__graph is not None and old_position in self.__graph:
            networkx.relabel_nodes(self.__graph, mapping, copy=False)
            if old_position in self.__graph:
                return False

    def has_links(self) -> bool:
        """Returns True if the graph is not None."""
        return self.__graph is not None

    def set_links(self, graph: Graph):
        if graph is None:  # Prevent accidental removal of data
            raise ValueError("Graph cannot be None. To remove links, use the remove_links method")
        self.__graph = graph

    def add_particle(self, particle: Particle):
        """Adds the particle as a node to the linking graphs. Does nothing if there is no linking data."""
        if self.__graph is not None:
            self.__graph.add_node(particle)
