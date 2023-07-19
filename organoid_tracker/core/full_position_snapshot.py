from typing import List, Dict, NamedTuple

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.typing import DataType


class FullPositionSnapshot(NamedTuple):
    """Represents a snapshot of all information of a position in a single class. This is useful if you want to restore
    a previous state."""

    @staticmethod
    def from_position(experiment: Experiment, position: Position) -> "FullPositionSnapshot":
        """Gets a snapshot of all information of a position."""
        links = list(experiment.links.find_links_of(position))
        connections = list(experiment.connections.find_connections(position))
        data = dict(experiment.position_data.find_all_data_of_position(position))
        return FullPositionSnapshot(position=position, links=links, connections=connections, position_data=data)

    @staticmethod
    def just_position(position: Position) -> "FullPositionSnapshot":
        """Creates a particle for a position that has no known shape, no links and no data."""
        return FullPositionSnapshot(position=position, links=list(), connections=list(), position_data=dict())

    @staticmethod
    def position_with_links(position: Position, *, links: List[Position]) -> "FullPositionSnapshot":
        """Creates a particle for a position that has no known data, but that has links to the given
        positions."""
        return FullPositionSnapshot(position=position, links=links, connections=list(), position_data=dict())

    position: Position
    links: List[Position]
    connections: List[Position]
    position_data: Dict[str, DataType]

    def restore(self, experiment: Experiment):
        """Restores a position, its shape, its metadata and its links."""
        experiment.positions.add(self.position)
        for link in self.links:
            experiment.links.add_link(self.position, link)
        for connection in self.connections:
            experiment.connections.add_connection(self.position, connection)
        for data_name, data_value in self.position_data.items():
            experiment.position_data.set_position_data(self.position, data_name, data_value)
