from enum import Enum
from typing import Optional

import networkx
from networkx import Graph

from autotrack.core.particles import Particle


class LinkType(Enum):
    """Type of linking network. Baseline is data that is assumed to be correct, scratch data can be used for comparison
    purposes."""
    SCRATCH = 1
    BASELINE = 2


class ParticleLinks:
    """Represents all particle links. Two different networks can be specified (called baseline and scratch), so that
    comparisons become possible. Care has been taken to ensure that the node sets of both linking networks are
    equal, so that comparisons between the networks are easier."""

    _scratch: Optional[Graph] = None
    _baseline: Optional[Graph] = None

    def add_links(self, type: LinkType, graph: Graph):
        """Adds all links from the graph. Existing link are not removed."""
        if type == LinkType.BASELINE:
            if self._baseline is None:
                self._baseline = graph
            else:
                self._baseline.add_nodes_from(graph.nodes)
                self._baseline.add_edges_from(graph.edges)
            self._synchronize_nodes()
        elif type == LinkType.SCRATCH:
            if self._scratch is None:
                self._scratch = graph
            else:
                self._scratch.add_nodes_from(graph.nodes)
                self._scratch.add_edges_from(graph.edges)
            self._synchronize_nodes()
        else:
            raise ValueError("Unknown link type: " + str(type))

    def remove_links(self, type: LinkType):
        """Removes all links of the specified type."""
        if type == LinkType.BASELINE:
            self._baseline = None
        elif type == LinkType.SCRATCH:
            self._scratch = None
        else:
            raise ValueError("Unknown link type: " + str(type))

    def remove_all_links(self):
        """Removes all links in the experiment."""
        self._baseline = None
        self._scratch = None

    @property
    def scratch(self) -> Optional[Graph]:
        """Links that are not 100% reliable."""
        return self._scratch

    @property
    def baseline(self) -> Optional[Graph]:
        """Links that are assumed to be reliable."""
        return self._baseline

    def remove_links_of_particle(self, particle: Particle):
        """Removes all links from and to the particle."""
        if self._scratch is not None and particle in self._scratch:
            self._scratch.remove_node(particle)
        if self._baseline is not None and particle in self._baseline:
            self._baseline.remove_node(particle)

    def replace_particle(self, old_position: Particle, position_new: Particle):
        """Replaces one particle with another. The old particle is removed from the graph, the new one is added. All
        links will be moved over to the new particle"""
        mapping = {old_position: position_new}
        if self._scratch is not None and old_position in self._scratch:
            networkx.relabel_nodes(self._scratch, mapping, copy=False)
            if old_position in self._scratch:
                return False
        if self._baseline is not None and old_position in self._baseline:
            networkx.relabel_nodes(self._baseline, mapping, copy=False)
            if old_position in self._baseline:
                return False

    def get_baseline_else_scratch(self) -> Optional[Graph]:
        """Gets the baseline links, if available. Else this method returns the scratch links, if available. If those
        also aren't available, this method returns None."""
        if self._baseline is not None:
            return self._baseline
        return self._scratch

    def get_scratch_else_baseline(self):
        """Gets the scratch links, if available. Else this method returns the baseline links, if available. If those
        also aren't available, this method returns None."""
        if self._scratch is not None:
            return self._scratch
        return self._baseline

    def can_compare_links(self) -> bool:
        """Returns True if both the scratch and baseline links are present, which makes comparisons between the two
        possible."""
        return self._scratch is not None and self._baseline is not None

    def has_links(self) -> bool:
        """Returns True if scratch and/or baseline links are present. If this method returns True, then the
        get_scratch_else_baseline and get_baseline_else_scratch methods will return a linking graph."""
        return self._scratch is not None or self._baseline is not None

    def _synchronize_nodes(self):
        """Keeps the node sets of both graphs equal, which is necessary for comparison purposes."""
        if not self.can_compare_links():
            return
        self._scratch.add_nodes_from(self._baseline)
        self._baseline.add_nodes_from(self._scratch)

    def set_links(self, type: LinkType, graph: Graph):
        if graph is None:  # Prevent accidental removal of data
            raise ValueError("Graph cannot be None. To remove links, use the remove_links method")
        if type == LinkType.SCRATCH:
            self._scratch = graph
            self._synchronize_nodes()
        elif type == LinkType.BASELINE:
            self._baseline = graph
            self._synchronize_nodes()
        else:
            raise ValueError("Unknown link type: " + str(type))
