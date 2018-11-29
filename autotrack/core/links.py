from enum import Enum
from typing import Optional, Dict, Iterable, List, AbstractSet, Set, Union, Tuple

import networkx
from networkx import Graph

from autotrack.core.particles import Particle


class ParticleLinks:
    """Represents all particle links. Two different networks can be specified (called baseline and scratch), so that
    comparisons become possible. Care has been taken to ensure that the node sets of both linking networks are
    equal, so that comparisons between the networks are easier."""

    _graph: Optional[Graph]

    def __init__(self, graph: Graph = None):
        self._graph = graph

    def add_links(self, links: "ParticleLinks"):
        """Adds all links from the graph. Existing link are not removed."""
        if self._graph is None:
            self._graph = links._graph
        else:
            self._graph = networkx.compose(links._graph, self._graph)

    def remove_all_links(self):
        """Removes all links in the experiment."""
        self._graph = None

    def remove_links_of_particle(self, particle: Particle):
        """Removes all links from and to the particle."""
        if self._graph is not None and particle in self._graph:
            self._graph.remove_node(particle)

    def replace_particle(self, old_position: Particle, position_new: Particle):
        """Replaces one particle with another. The old particle is removed from the graph, the new one is added. All
        links will be moved over to the new particle"""
        mapping = {old_position: position_new}
        if self._graph is not None and old_position in self._graph:
            networkx.relabel_nodes(self._graph, mapping, copy=False)
            if old_position in self._graph:
                return False

    def has_links(self) -> bool:
        """Returns True if the graph is not None."""
        return self._graph is not None

    def add_particle(self, particle: Particle):
        """Adds the particle as a node to the linking graphs. Initialized the linking graph if necessary."""
        if self._graph is None:
            self._graph = Graph()
        self._graph.add_node(particle)

    def to_d3_data(self) -> Dict:
        """Return data in D3.js node-link format that is suitable for JSON serialization
        and use in Javascript documents."""
        if self._graph is None:
            return {}
        return networkx.node_link_data(self._graph)

    def add_d3_data(self, data: Dict):
        """Adds data in the D3.js node-link format. Used for deserialization."""
        graph = networkx.node_link_graph(data)
        if self._graph is None:
            self._graph = graph
        else:
            self._graph = networkx.compose(graph, self._graph)

    def find_futures(self, particle: Particle) -> Set[Particle]:
        """Returns all connections to the future."""
        if self._graph is None:
            return set()
        if particle not in self._graph:
            return set()
        linked_particles = self._graph[particle]
        return {linked_particle for linked_particle in linked_particles
                if linked_particle.time_point_number() > particle.time_point_number()}

    def find_pasts(self, particle: Particle) -> Set[Particle]:
        """Returns all connections to the past."""
        if self._graph is None:
            return set()
        if particle not in self._graph:
            return set()
        linked_particles = self._graph[particle]
        return {linked_particle for linked_particle in linked_particles
                if linked_particle.time_point_number() < particle.time_point_number()}

    def find_appeared_cells(self, time_point_number_to_ignore: Optional[int] = None) -> Iterable[Particle]:
        """This method gets all particles that "popped up out of nothing": that have no links to the past. You can give
        this method a time point number to ignore. Usually, this would be the first time point number of the experiment,
        as cells that have no links to the past in the first time point are not that interesting."""
        if self._graph is None:
            return []

        for particle in self._graph.nodes():
            if len(self.find_pasts(particle)) == 0:
                if time_point_number_to_ignore is None or time_point_number_to_ignore != particle.time_point_number():
                    yield particle

    def add_link(self, particle1: Particle, particle2: Particle):
        """Adds a link between the particles. The linking network will be initialized if necessary."""
        if self._graph is None:
            self._graph = Graph()
        self._graph.add_node(particle1)
        self._graph.add_node(particle2)
        self._graph.add_edge(particle1, particle2)

    def get_particle_data(self, particle: Particle, data_name: str) -> Union[str, int, None]:
        """Gets the attribute of the particle with the given name. Returns None if not found."""
        if self._graph is None:
            return None
        data = self._graph.nodes.get(particle)
        if data is None:
            return None
        return data.get(data_name)

    def set_particle_data(self, particle: Particle, data_name: str, value: Union[str, int, None]):
        """Adds or overwrites the given attribute for the given particle. Set value to None to delete the attribute.

        Note: this is a low-level API. See the linking_markers module for more high-level methods, for example for how
        to read end markers, error markers, etc.
        """
        if value is None:
            if self._graph is None:
                return  # No links, so nothing to delete
            try:
                del self._graph.nodes[particle][data_name]
            except KeyError:
                pass  # Ignore, nothing to delete
            return

        if self._graph is None:
            self._graph = Graph()

        # Next line of code has some complex syntax to support dynamic attribute names.
        # If data_name == "foo", then the line is equal to self.__graph.add_node(particle, foo=value)
        self._graph.add_node(particle, **{data_name: value})

    def find_links_of(self, particle: Particle) -> Iterable[Particle]:
        """Gets all links of a particle, both to the past and the future."""
        if self._graph is None:
            return []  # No graph
        try:
            return self._graph[particle]
        except KeyError:
            return []  # Particle not in graph

    def find_all_particles(self) -> Iterable[Particle]:
        """Gets all particles in the linking graph. Note that particles without links are not included here."""
        if self._graph is None:
            return []
        return self._graph.nodes()

    def remove_link(self, particle1: Particle, particle2: Particle):
        """Removes the link between the given particles. Does nothing if there is no link between the particles."""
        if self._graph is None:
            return
        if self._graph.has_edge(particle1, particle2):
            self._graph.remove_edge(particle1, particle2)

    def contains_link(self, particle1: Particle, particle2: Particle) -> bool:
        """Returns True if the two given particles are linked to each other."""
        if self._graph is None:
            return False
        return self._graph.has_edge(particle1, particle2)

    def contains_particle(self, particle: Particle) -> bool:
        """Returns True if the given particle is part of this linking network."""
        if self._graph is None:
            return False
        return particle in self._graph

    def find_all_links(self) -> Iterable[Tuple[Particle, Particle]]:
        """Gets all available links."""
        if self._graph is None:
            return []
        return self._graph.edges

    def copy(self) -> "ParticleLinks":
        """Returns a copy of all the links, so that you can modify that data set without affecting this one."""
        copy = ParticleLinks()
        if self._graph is not None:
            copy._graph = self._graph.copy()
        return copy

    def limit_to_time_points(self, first_time_point_number: int, last_time_point_number: int) -> "ParticleLinks":
        """Returns a view of the links consisting of only the particles between the first and last time point number,
        inclusive."""
        if self._graph is None:
            return self

        def _is_in_time_points(particle: Particle) -> bool:
            return first_time_point_number <= particle.time_point_number() <= last_time_point_number

        subgraph = self._graph.subgraph([particle for particle in self._graph.nodes()
                                         if _is_in_time_points(particle)])
        return ParticleLinks(subgraph)
