"""Additional metadata of a position, like the cell type or the fluorescent intensity."""
from typing import Set, Dict, Optional, Iterable, Union, List, Tuple

import numpy
from numpy import ndarray

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData


def get_position_type(position_data: PositionData, position: Position) -> Optional[str]:
    """Gets the type of the cell in UPPERCASE, interpreted as the intestinal organoid cell type."""
    type = position_data.get_position_data(position, "type")
    if type is None:
        return None
    return type.upper()


def set_position_type(position_data: PositionData, position: Position, type: Optional[str]):
    """Sets the type of the cell. Set to None to delete the cell type."""
    type_str = type.upper() if type is not None else None
    position_data.set_position_data(position, "type", type_str)


def get_position_types(position_data: PositionData, positions: Set[Position]) -> Dict[Position, Optional[str]]:
    """Gets all known cell types of the given positions, with the names in UPPERCASE."""
    types = dict()
    for position in positions:
        types[position] = get_position_type(position_data, position)
    return types


def get_positions_of_type(position_data: PositionData, requested_type: str) -> Iterable[Position]:
    """Gets all positions of the requested cell type."""
    requested_type = requested_type.upper()
    return (position for position, position_type in position_data.find_all_positions_with_data("type")
            if position_type.upper() == requested_type)


def set_raw_intensities(experiment: Experiment, raw_intensities: Dict[Position, int], volumes: Dict[Position, int]):
    """Registers the given intensities for the given positions. Both dicts must have the same keys.

    Also removes any previously set intensity normalization."""
    if raw_intensities.keys() != volumes.keys():
        raise ValueError("Need to supply intensities and volumes for the same cells")
    experiment.position_data.add_positions_data("intensity", raw_intensities)
    experiment.position_data.add_positions_data("intensity_volume", volumes)
    remove_intensity_normalization(experiment)


def get_raw_intensity(position_data: PositionData, position: Position) -> Optional[float]:
    """Gets the raw intensity of the position."""
    return position_data.get_position_data(position, "intensity")


def get_normalized_intensity(experiment: Experiment, position: Position) -> Optional[float]:
    """Gets the normalized intensity of the position."""
    position_data = experiment.position_data
    global_data = experiment.global_data

    intensity = position_data.get_position_data(position, "intensity")
    background = global_data.get_data("intensity_background_per_pixel")
    multiplier = global_data.get_data("intensity_multiplier_z" + str(round(position.z)))
    if multiplier is None:
        # Try global multiplier
        multiplier = global_data.get_data("intensity_multiplier")
    volume = position_data.get_position_data(position, "intensity_volume")
    if volume is None or multiplier is None or background is None:
        return intensity
    return (intensity - background * volume) * multiplier


def perform_intensity_normalization(experiment: Experiment, *, background_correction: bool = True, z_correction: bool = True):
    """Gets the average intensity of all positions in the experiment.
    Returns None if there are no intensity recorded."""
    remove_intensity_normalization(experiment)

    intensities = list()
    volumes = list()
    zs = list()

    position_data = experiment.position_data
    for position, intensity in position_data.find_all_positions_with_data("intensity"):
        volume = position_data.get_position_data(position, "intensity_volume")
        if volume is None and background_correction:
            continue

        intensities.append(intensity)
        volumes.append(volume)
        zs.append(round(position.z))

    if len(intensities) == 0:
        return

    intensities = numpy.array(intensities, dtype=numpy.float32)
    volumes = numpy.array(volumes, dtype=numpy.float32)
    zs = numpy.array(zs, dtype=numpy.int32)

    if background_correction:
        # Assume the lowest signal consists of only background
        lowest_intensity_index = numpy.argmin(intensities / volumes)
        background_per_px = float(intensities[lowest_intensity_index] / volumes[lowest_intensity_index])

        # Subtract this background
        intensities -= volumes * background_per_px
        experiment.global_data.set_data("intensity_background_per_pixel", background_per_px)
    else:
        experiment.global_data.set_data("intensity_background_per_pixel", 0)

    # Now normalize the mean to 100
    if z_correction:
        for z in range(int(numpy.min(zs)), int(numpy.max(zs)) + 1):
            median = numpy.median(intensities[zs == z])
            normalization_factor = float(100 / median)
            experiment.global_data.set_data("intensity_multiplier_z" + str(z), normalization_factor)
    else:
        median = numpy.median(intensities)
        normalization_factor = float(100 / median)
        experiment.global_data.set_data("intensity_multiplier", normalization_factor)


def remove_intensity_normalization(experiment: Experiment):
    """Removes the normalization set by perform_intensity_normalization."""
    experiment.global_data.set_data("intensity_background_per_pixel", None)
    experiment.global_data.set_data("intensity_multiplier", None)
    for key in list(experiment.global_data.get_all_data().keys()):
        if key.startswith("intensity_multiplier_z"):
            experiment.global_data.set_data(key, None)
