"""Additional metadata of a position, like the cell type or the fluorescent intensity."""
from typing import Set, Dict, Optional, Iterable

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_data import PositionData


# If a position has position data with this name, the error checker will always flag it for manual review. Useful for
# reminding yourself to revisit a position.
UNCERTAIN_MARKER = "uncertain"


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


def get_all_position_types(position_data: PositionData) -> Set[str]:
    """Gets a set of all position types that were used in the experiment."""
    found_types = set()
    for position, position_type in position_data.find_all_positions_with_data("type"):
        found_types.add(position_type.upper())
    return found_types


def get_positions_of_type(position_data: PositionData, requested_type: str) -> Iterable[Position]:
    """Gets all positions of the requested cell type."""
    requested_type = requested_type.upper()
    return (position for position, position_type in position_data.find_all_positions_with_data("type")
            if position_type.upper() == requested_type)


def set_raw_intensities(experiment: Experiment, raw_intensities: Dict[Position, int], volumes: Dict[Position, int]):
    """@deprecated Old method, please use intensity_calculator.set_raw_intensities instead."""
    from . import intensity_calculator
    intensity_calculator.set_raw_intensities(experiment, raw_intensities, volumes)


def get_raw_intensity(position_data: PositionData, position: Position) -> Optional[float]:
    """@deprecated Old method, please use intensity_calculator.get_raw_intensity instead."""
    from . import intensity_calculator
    return intensity_calculator.get_raw_intensity(position_data, position)


def get_normalized_intensity(experiment: Experiment, position: Position) -> Optional[float]:
    """@deprecated Old method, please use intensity_calculator.get_normalized_intensity instead."""
    from . import intensity_calculator
    return intensity_calculator.get_normalized_intensity(experiment, position)


def get_position_flags(experiment: Experiment) -> Iterable[str]:
    """Gets all used position flags of the experiment. These are simply all keys in experiment.position_data with the
    bool data type. In addition, the special flag `UNCERTAIN_MARKER` is always returned, as that flag is used in the
    error checker."""
    returned_uncertain_marker = False
    for data_name, data_type in experiment.position_data.get_data_names_and_types().items():
        if data_type == bool:
            yield data_name
            if data_name == UNCERTAIN_MARKER:
                returned_uncertain_marker = True

    # Always make sure that this one is returned, this flag is present by default
    if not returned_uncertain_marker:
        yield UNCERTAIN_MARKER
