from typing import List, Dict

from autotrack.core.experiment import Experiment
from autotrack.core.position import Position
from autotrack.core.shape import ParticleShape, UnknownShape
from autotrack.core.typing import DataType


class Particle:
    """Represents a snapshot of all information of a particle in a single class. This is useful if you want to restore
    a previous state."""

    @staticmethod
    def from_position(experiment: Experiment, position: Position) -> "Particle":
        """Gets a snapshot of all information of a position."""
        links = list(experiment.links.find_futures(position)) + list(experiment.links.find_pasts(position))
        shape = experiment.positions.get_shape(position)
        data = dict(experiment.links.find_all_data_of_position(position))
        return Particle(position, shape, links, data)

    @staticmethod
    def just_position(position: Position) -> "Particle":
        """Creates a particle for a position that has no known shape, no links and no data."""
        return Particle(position, UnknownShape(), list(), dict())

    @staticmethod
    def position_with_links(position: Position, *, links: List[Position]) -> "Particle":
        """Creates a particle for a position that has no known shape and no data, but that has links to the given
        positions."""
        return Particle(position, UnknownShape(), links, dict())

    position: Position
    shape: ParticleShape
    links: List[Position]
    data: Dict[str, DataType]

    def __init__(self, position: Position, shape: ParticleShape, links: List[Position], data: Dict[str, DataType]):
        self.position = position
        self.shape = shape
        self.links = links
        self.data = data

    def restore(self, experiment: Experiment):
        """Restores a position, its shape, its metadata and its links."""
        experiment.positions.add(self.position, self.shape)
        for link in self.links:
            experiment.links.add_link(self.position, link)
        for data_name, data_value in self.data.items():
            experiment.links.set_position_data(self.position, data_name, data_value)

    def __repr__(self) -> str:
        return "Particle(" + repr(self.position) + ", " + repr(self.shape) + ", " + repr(self.links) + ", " \
               + repr(self.data) + ")"
