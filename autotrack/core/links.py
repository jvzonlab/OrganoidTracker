from enum import Enum
from typing import Optional, Dict, Iterable, List, AbstractSet, Set, Union

import networkx
from networkx import Graph

from autotrack.core.particles import Particle


class ParticleLinks:
    """Represents all particle links. Two different networks can be specified (called baseline and scratch), so that
    comparisons become possible. Care has been taken to ensure that the node sets of both linking networks are
    equal, so that comparisons between the networks are easier."""

    __graph: Optional[Graph]

    def __init__(self, graph: Graph = None):
        self.__graph = graph

    def add_links(self, links: Union[Graph, "ParticleLinks"]):
        """Adds all links from the graph. Existing link are not removed."""
        graph = links if isinstance(links, Graph) else links.graph

        if self.__graph is None:
            self.__graph = graph
        else:
            self.__graph = networkx.compose(graph, self.__graph)

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

    def set_links(self, links: Union[Graph, "ParticleLinks"]):
        if links is None:  # Prevent accidental removal of data
            raise ValueError("Graph cannot be None. To remove links, use the remove_links method")
        graph = links if isinstance(links, Graph) else links.graph
        self.__graph = graph

    def add_particle(self, particle: Particle):
        """Adds the particle as a node to the linking graphs. Initialized the linking graph if necessary."""
        if self.__graph is None:
            self.__graph = Graph()
        self.__graph.add_node(particle)

    def node_link_data(self) -> Dict:
        """Return data in node-link format that is suitable for JSON serialization
        and use in Javascript documents."""
        if self.__graph is None:
            return {}
        return networkx.node_link_data(self.__graph)

    def find_futures(self, particle: Particle) -> Set[Particle]:
        """Returns all connections to the future."""
        if self.__graph is None:
            return set()
        if particle not in self.__graph:
            return set()
        linked_particles = self.__graph[particle]
        return {linked_particle for linked_particle in linked_particles
                if linked_particle.time_point_number() > particle.time_point_number()}

    def find_pasts(self, particle: Particle) -> Set[Particle]:
        """Returns all connections to the past."""
        if self.__graph is None:
            return set()
        if particle not in self.__graph:
            return set()
        linked_particles = self.__graph[particle]
        return {linked_particle for linked_particle in linked_particles
                if linked_particle.time_point_number() < particle.time_point_number()}

    def find_appeared_cells(self, time_point_number_to_ignore: Optional[int] = None) -> Iterable[Particle]:
        """This method gets all particles that "popped up out of nothing": that have no links to the past. You can give
        this method a time point number to ignore. Usually, this would be the first time point number of the experiment,
        as cells that have no links to the past in the first time point are not that interesting."""
        if self.__graph is None:
            return []

        for particle in self.__graph.nodes():
            if len(self.find_pasts(particle)) == 0:
                if time_point_number_to_ignore is None or time_point_number_to_ignore != particle.time_point_number():
                    yield particle

    def add_link(self, particle1: Particle, particle2: Particle):
        """Adds a link between the particles. The linking network will be initialized if necessary."""
        if self.__graph is None:
            self.__graph = Graph()
        self.__graph.add_node(particle1)
        self.__graph.add_node(particle2)
        self.__graph.add_edge(particle1, particle2)

    def get_particle_data(self, particle: Particle, data_name: str) -> Union[str, int, None]:
        """Gets the attribute of the particle with the given name. Returns None if not found."""
        if self.__graph is None:
            return None
        data = self.__graph.nodes.get(particle)
        if data is None:
            return None
        return data.get(data_name)

    def set_particle_data(self, particle: Particle, data_name: str, value: Union[str, int, None]):
        """Adds or overwrites the given attribute for the given particle. Set value to None to delete the attribute.

        Note: this is a low-level API. See the linking_markers module for more high-level methods, for example for how
        to read end markers, error markers, etc.
        """
        if value is None:
            if self.__graph is None:
                return  # No links, so nothing to delete
            try:
                del self.__graph.nodes[particle][data_name]
            except KeyError:
                pass  # Ignore, nothing to delete
            return

        if self.__graph is None:
            self.__graph = Graph()

        # Next line of code has some complex syntax to support dynamic attribute names.
        # If data_name == "foo", then the line is equal to self.__graph.add_node(particle, foo=value)
        self.__graph.add_node(particle, **{data_name: value})

    def find_links_of(self, particle: Particle) -> Iterable[Particle]:
        """Gets all links of a particle, both to the past and the future."""
        if self.__graph is None:
            return []  # No graph
        try:
            return self.__graph[particle]
        except KeyError:
            return []  # Particle not in graph

    def find_all_particles(self) -> Iterable[Particle]:
        """Gets all particles in the linking graph. Note that particles without links are not included here."""
        if self.__graph is None:
            return []
        return self.__graph.nodes()

    def remove_link(self, particle1: Particle, particle2: Particle):
        """Removes the link between the given particles. Does nothing if there is no link between the particles."""
        if self.__graph is None:
            return
        if self.__graph.has_edge(particle1, particle2):
            self.__graph.remove_edge(particle1, particle2)

    def has_link(self, particle1: Particle, particle2: Particle) -> bool:
        """Returns True if the two given particles are linked to each other."""
        if self.__graph is None:
            return False
        return self.__graph.has_edge(particle1, particle2)
