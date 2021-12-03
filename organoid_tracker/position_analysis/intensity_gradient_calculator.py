from typing import Optional, Callable, List

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.vector import Vector3
from organoid_tracker.position_analysis import position_markers


def get_intensity_gradient(experiment: Experiment, position: Position, is_included: Callable[[Position], bool]
                           ) -> Optional[Vector3]:
    """Calculates the spatial intensity gradient for a position. Note: doesn't check whether all neighbors are present."""
    resolution = experiment.images.resolution()
    position_intensity = position_markers.get_normalized_intensity(experiment, position)

    neighbors = list(position for position in experiment.connections.find_connections(position)
                     if is_included(position))
    neighbor_positions = [neighbor.to_vector_um(resolution)
                            for neighbor in neighbors]
    neighbor_intensities = [position_markers.get_normalized_intensity(experiment, neighbor)
                            for neighbor in neighbors]

    if None in neighbor_intensities:
        return None

    result = _calculate_intensity_gradient(position.to_vector_um(resolution), neighbor_positions, position_intensity, neighbor_intensities)

    return result


def _calculate_intensity_gradient(position_coi: Vector3, positions_neighbors: List[Vector3],
                                  intensity_coi: float, intensities_neighbors: List[float]):
    """

    Parameters
    ----------
    position_coi : Vector3
        Positions of the cell of interest (coi).
    positions_neighbors : List of N Vector3
        Positions of the N neighbors of the coi.
    intensity_coi : float
        Intensity of coi.
    intensities_neighbors : numpy ndarray of length N
        Intensities of all N neighbors of coi.

    Returns
    -------
    Vector pointing in the direction of the gradient experienced by cell of interest.
    """
    number_of_neighbors = len(positions_neighbors)

    # Calculate gradient-entries
    gradient_vector = Vector3.sum(
        ((position_neighbor - position_coi).normalized().multiply(intensity_neighbor - intensity_coi).divide(number_of_neighbors)
                   for intensity_neighbor, position_neighbor in zip(intensities_neighbors, positions_neighbors))
    )

    return gradient_vector
