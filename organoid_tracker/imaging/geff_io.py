import os
from enum import auto, Enum
from typing import Any, List, NamedTuple, Optional, Dict, Type
from typing import Literal

import geff
import numpy
import zarr
from geff import GeffReader
from geff._typing import ZarrPropDict, PropDictNpArray, InMemoryGeff
from geff.core_io import write_arrays
from geff.core_io._serialization import deserialize_vlen_property_data
from geff_spec import Axis, GeffMetadata
from statsmodels.tsa.adfvalues import z_ct_smallp
from zarr.storage import StoreLike

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.core.typing import DataType

# Multiply your value by these factors to convert to micrometers
_MULTIPLICATION_FACTOR_TO_MICROMETERS = {
    "meter": 1e6,
    "centimeter": 1e4,
    "millimeter": 1e3,
    "micrometer": 1.0,
    "nanometer": 1e-3,
}
_MULTIPLICATION_FACTOR_TO_MINUTES = {
    "day": 1440.0,
    "hour": 60.0,
    "minute": 1.0,
    "second": 1/60.0,
    "millisecond": 1/60000.0,
}


class _OurAxisType(Enum):
    TIME = auto()
    X = auto()
    Y = auto()
    Z = auto()


class _AxisInfo:
    scale: float  = 1  # Whatever scale factor we need to apply to get from
    offset: float = 0
    axis_name: str = "none"


class _AxesInfo:
    x: _AxisInfo
    y: _AxisInfo
    z: _AxisInfo
    time: _AxisInfo

    def __init__(self):
        self.x = _AxisInfo()
        self.y = _AxisInfo()
        self.z = _AxisInfo()
        self.time = _AxisInfo()

    def axis(self, axis_type: _OurAxisType) -> _AxisInfo:
        if axis_type == _OurAxisType.TIME:
            return self.time
        elif axis_type == _OurAxisType.X:
            return self.x
        elif axis_type == _OurAxisType.Y:
            return self.y
        elif axis_type == _OurAxisType.Z:
            return self.z
        raise ValueError(f"Unknown axis type: {axis_type}")


def _read_prop(group: zarr.Group, name: str, prop_type: Literal["node", "edge"]) -> ZarrPropDict:
    """Read a single property into a zarr property dictionary."""
    group_path = f"nodes/props/{name}" if prop_type == "node" else f"edges/props/{name}"
    prop_group = zarr.open_group(group.store, path=group_path, mode="r")
    values: zarr.Array = prop_group.get("values")
    prop_dict: ZarrPropDict = {"values": values}
    if "missing" in prop_group.keys():
        missing = prop_group.get("missing")
        prop_dict["missing"] = missing

    if "data" in prop_group.keys():
        prop_dict["data"] = prop_group.get("data")
    return prop_dict


def load_data_file(geff_source: str, min_time_point: int = -9999999, max_time_point: int = 9999999, experiment: Experiment = None):
    experiment = experiment if experiment is not None else Experiment()

    # Correct for cases where the path doesn't end with .geff (because the user selected a file inside)
    if isinstance(geff_source, str) and f".geff{os.sep}" in geff_source:
        geff_index = geff_source.lower().rindex(".geff")
        geff_source = geff_source[:geff_index + 5]

    geff_reader = GeffReader(geff_source)
    geff_reader.read_node_props()
    geff_reader.read_edge_props()
    in_memory_geff = geff_reader.build()
    del geff_reader  # Free resources

    # Read in the metadata names
    node_prop_names = list()
    for node_prop_info in in_memory_geff["metadata"].node_props_metadata.values():
        # Name, description and unit not handled yet
        identifier = node_prop_info.identifier
        if identifier in ["t", "x", "y", "z"]:
            continue  # Skip position properties
        node_prop_names.append(identifier)
    edge_prop_names = list()
    for edge_prop_info in in_memory_geff["metadata"].edge_props_metadata.values():
        edge_prop_names.append(edge_prop_info.identifier)

    # Read in the positions
    positions_by_node_id = _read_positions(experiment, in_memory_geff, min_time_point, max_time_point, node_prop_names)

    # Read in the edges
    edge_ids = in_memory_geff["edge_ids"]
    edge_props = in_memory_geff["edge_props"]
    experiment_links = experiment.links
    for i, (source_node_id, target_node_id) in enumerate(edge_ids):
        source_position = positions_by_node_id[source_node_id]
        target_position = positions_by_node_id[target_node_id]
        if source_position is None or target_position is None:
            continue  # Outside of time point range - no position was created
        experiment_links.add_link(source_position, target_position)

        # Read in the edge metadata
        for edge_prop_name in edge_prop_names:
            prop_array = edge_props[edge_prop_name]
            prop_values = prop_array["values"]
            prop_missing = prop_array.get("missing")
            if prop_missing is not None and prop_missing[i]:
                continue  # Missing value
            experiment_links.set_link_data(source_position, target_position, edge_prop_name, prop_values[i].tolist())

    # Read in any extra metadata
    geff_metadata: GeffMetadata = in_memory_geff["metadata"]
    for key, value in geff_metadata.extra.items():
        if isinstance(value, numpy.generic):
            value = value.tolist()  # Convert numpy scalar to native Python type

        experiment.global_data.set_data(key, value)

    return experiment


def _read_positions(experiment: Experiment, in_memory_geff: InMemoryGeff, min_time_point: int, max_time_point: int,
                    node_prop_names: list[str]) -> list[Position | None]:
    """Read the positions and position metadata from the in-memory GEFF data and adds them to the experiment.
    Returns a list of positions by node id."""

    # Read in the scale factors
    geff_axes = in_memory_geff["metadata"].axes
    axes_info = _read_axes_info(experiment, geff_axes)

    node_ids = in_memory_geff["node_ids"]
    node_props = in_memory_geff["node_props"]

    time_values = node_props[axes_info.time.axis_name]["values"]
    x_values = node_props[axes_info.x.axis_name]["values"]
    y_values = node_props[axes_info.y.axis_name]["values"]
    if axes_info.z.axis_name in node_props:
        z_values = node_props[axes_info.z.axis_name]["values"]
    else:
        z_values = numpy.zeros_like(x_values)  # For 2D-tracking, just fill in zeros for the Z

    if numpy.min(time_values) == numpy.max(time_values) == 0:
        # Fix for a malformed GEFF file of 2D cell tracking, where the Z and T axis were swapped
        time_values = z_values
        z_values = numpy.zeros_like(time_values)

    positions_by_node_id = []
    experiment_positions = experiment.positions
    for i in range(len(node_ids)):
        node_id = node_ids[i]

        time_point_number = int(time_values[i])
        if time_point_number < min_time_point or time_point_number > max_time_point:
            # Skip positions outside the requested time point range
            continue

        position = Position(float(x_values[i] * axes_info.x.scale + axes_info.x.offset),
                            float(y_values[i] * axes_info.y.scale + axes_info.y.offset),
                            float(z_values[i] * axes_info.z.scale + axes_info.z.offset),
                            time_point_number=int(time_values[i] * axes_info.time.scale + axes_info.time.offset),)

        if node_id == len(positions_by_node_id):
            # Most common case - sequential node ids
            positions_by_node_id.append(position)
        else:
            # Handle non-sequential node ids - append the list with None until it's big enough
            while len(positions_by_node_id) <= node_id:
                positions_by_node_id.append(None)
            positions_by_node_id[node_id] = position
        experiment_positions.add(position)

        # Read in the position metadata
        for node_prop_name in node_prop_names:
            prop_array = node_props[node_prop_name]
            prop_values = prop_array["values"]
            prop_missing = prop_array["missing"]
            if prop_missing is not None and prop_missing[i]:
                continue  # Missing value
            experiment_positions.set_position_data(position, node_prop_name, prop_values[i].tolist())
    return positions_by_node_id


def _read_axes_info(experiment: Experiment, geff_axes: list[Axis]) -> _AxesInfo:
    """Read the scale factors for time, z, y, x axes from the GEFF axes. Raises an UserError if unsupported units are found."""
    our_axes_info = _AxesInfo()

    for ax in geff_axes:

        # Read any extra scale
        original_unit = ax.unit
        final_unit = ax.unit
        scale_factor = 1
        offset = ax.offset if "offset" in ax else 0
        if "scale" in ax:
            scale_factor = ax.scale
            final_unit = ax.scaled_unit  # scaled_unit must exist if scale is set, according to the spec

        if ax.type == "time":
            our_axis_type = _OurAxisType.TIME

            if original_unit is None:
                original_unit = "frame"
            if final_unit is None:
                final_unit = "frame"

            if final_unit != "frame":
                # Need to adjust the scale so that we end up with frames
                if original_unit == "frame":
                    # Easy - original was in time frames, so just remove the scaling
                    offset /= scale_factor  # Offset is applied after scaling, so we need to adjust it as well
                    scale_factor = 1
                resolution = _get_resolution(experiment)
                if final_unit not in _MULTIPLICATION_FACTOR_TO_MINUTES:
                    raise UserError("Unsupported file",
                                    f"The unit '{final_unit}' for the time axis '{ax.name}' is not supported in our GEFF loader")
                scale_factor *= _MULTIPLICATION_FACTOR_TO_MINUTES[final_unit] / resolution.time_point_interval_m
                offset *= _MULTIPLICATION_FACTOR_TO_MINUTES[final_unit] / resolution.time_point_interval_m
        elif ax.type == "space":
            if not ax.name in ("x", "y", "z"):
                raise UserError("Unsupported file",
                                f"Spatial axis name '{ax.name}' is not supported in our GEFF loader. Only 'x', 'y' and 'z' are supported.")
            our_axis_type = _OurAxisType[ax.name.upper()]  # Works, because ax_name can only be x, y or z at this point

            if original_unit is None:
                original_unit = "pixel"
            if final_unit is None:
                final_unit = "pixel"
            if final_unit != "pixel":
                # Need to adjust the scale so that we end up with pixels
                if original_unit == "pixel":
                    # Easy, original was in pixels, so just don't apply the scaling
                    offset /= scale_factor  # Offset is applied after scaling, to adjust it as well
                    scale_factor = 1
                else:
                    # Make a calculation on how to go from this unit to pixels
                    resolution = _get_resolution(experiment)
                    if final_unit not in _MULTIPLICATION_FACTOR_TO_MICROMETERS:
                        raise UserError("Unsupported file",
                                        f"The unit '{final_unit}' for the {ax_name}-axis is not supported in our GEFF loader")
                    scale_to_micrometers = _MULTIPLICATION_FACTOR_TO_MICROMETERS[final_unit]
                    if ax.name == "x":
                        extra_scale_factor = scale_to_micrometers / resolution.pixel_size_x_um
                    elif ax.name == "y":
                        extra_scale_factor = scale_to_micrometers / resolution.pixel_size_y_um
                    elif ax.name == "z":
                        extra_scale_factor = scale_to_micrometers / resolution.pixel_size_z_um
                    else:
                        raise ValueError(ax.name)  # Shouldn't happen, we checked above that ax_name is x, y or z
                    scale_factor *= extra_scale_factor
                    offset *= extra_scale_factor  # Offset is applied after scaling, so adjust too

        elif ax.type == "channel" or ax.type is None:
            continue
        else:
            raise NotImplementedError(f"Axis type {ax.type} is not supported in our GEFF loader")

        # Put all collected info into axes_info
        our_axis_info = our_axes_info.axis(our_axis_type)
        our_axis_info.axis_name = ax.name
        our_axis_info.scale_factor = scale_factor
        our_axis_info.offset = offset

    return our_axes_info


def _get_resolution(experiment: Experiment) -> ImageResolution:
    try:
        resolution = experiment.images.resolution()
    except UserError:
        raise UserError("Spatial resolution error",
                        "GEFF file has positions in spatial units, and we need to convert them to pixel positions."
                        " However, we're missing the image resolution, so we cannot do this."
                        "\n\nUsually the resolution is set when loading the images, and otherwise, use 'Edit > Set image resolution...' to set it manually.")
    return resolution


class _InMemoryPropDictNpArray:
    data_type: Optional[Type[DataType]]
    values: List[Optional[DataType | numpy.ndarray]]

    __slots__ = ("data_type", "values", "missing")

    def __init__(self, data_type: Optional[Type[DataType]]):
        self.data_type = data_type
        self.values = []

    def _get_numpy_type(self) -> Type[numpy.generic]:
        """Returns the numpy type corresponding to the data type."""
        if self.data_type is None:
            example_value = next((v for v in self.values if v is not None), None)
            if example_value is None:
                return numpy.float64  # Default to float64 if all values are None
            elif isinstance(example_value, float):
                return numpy.float64
            elif isinstance(example_value, int):
                return numpy.int64
            elif isinstance(example_value, str):
                return numpy.str_
            elif isinstance(example_value, bool):
                return numpy.bool_
            else:
                return numpy.object_

        if self.data_type == int:
            return numpy.int64
        elif self.data_type == float:
            return numpy.float64
        elif self.data_type == str:
            return numpy.str_
        elif self.data_type == bool:
            return numpy.bool_
        else:
            return numpy.object_

    def _get_numpy_list_type(self) -> Optional[Type[numpy.generic]]:
        """In case we're storing lists of a specific type, return the type of elements in the list."""
        example_value = next((v for v in self.values if v), None)
        if isinstance(example_value, list):
            first_element = example_value[0]
            if isinstance(first_element, float):
                return numpy.float64
            elif isinstance(first_element, int):
                return numpy.int64
            elif isinstance(first_element, str):
                return numpy.str_
            elif isinstance(first_element, bool):
                return numpy.bool_

        return None  # Not a list of specific type



    def to_geff_prop_dict(self) -> PropDictNpArray:
        """Converts the data structures a dictionary with numpy arrays for GEFF serialization."""
        numpy_type = self._get_numpy_type()
        numpy_list_type = self._get_numpy_list_type()

        if numpy_list_type == numpy.str_:
            # Can't save lists of strings in GEFF unfortunately, just plain strings
            # So we need to convert ["foo", "bar"] to "['foo', 'bar']"
            for i, value in enumerate(self.values):
                if value is not None:
                    self.values[i] = str(value)
            numpy_list_type = None
            numpy_type = numpy.str_

        # Data conversions
        missing_array = None
        for i, value in enumerate(self.values):
            if value is None:
                if missing_array is None:
                    # Initialize missing array
                    missing_array = numpy.zeros(len(self.values), dtype=bool)
                missing_array[i] = True

                # Need to replace missing value with a default value
                if numpy_type == numpy.bool_:
                    self.values[i] = False
                elif numpy_type == numpy.int64:
                    self.values[i] = 0
                elif numpy_type == numpy.str_:
                    self.values[i] = ""
                elif numpy_type == numpy.float64:
                    self.values[i] = numpy.nan
                else:  # Arrays
                    self.values[i] = numpy.zeros((0,), dtype=numpy_list_type)

            # Convert Python lists to numpy arrays
            if isinstance(value, list):
                self.values[i] = numpy.array(value)

        values_array = numpy.array(self.values, dtype=numpy_type)

        return {
            "values": values_array,
            "missing": missing_array,
        }


def save_data_file(experiment: Experiment, file_name: str):
    """Saves the experiment tracking data to a GEFF file."""
    experiment_positions = experiment.positions
    experiment_links = experiment.links

    # Set up the arrays
    node_ids = []
    edge_ids = []
    node_props = dict()
    position_data_names = list()  # All position metadata names, but not x,y,z,t
    for node_prop_name, data_type in experiment_positions.get_data_names_and_types().items():
        node_props[node_prop_name] = _InMemoryPropDictNpArray(data_type)
        position_data_names.append(node_prop_name)
    edge_props = dict()
    for edge_prop_name in experiment_links.find_all_data_names():
        edge_props[edge_prop_name] = _InMemoryPropDictNpArray(None)

    # These metadata arrays are always present, they encode the position coordinates
    node_x_array, node_y_array, node_z_array, node_t_array = _InMemoryPropDictNpArray(float), _InMemoryPropDictNpArray(float), _InMemoryPropDictNpArray(float), _InMemoryPropDictNpArray(int)
    node_props["x"] = node_x_array
    node_props["y"] = node_y_array
    node_props["z"] = node_z_array
    node_props["t"] = node_t_array

    # Collect all the data
    current_time_point_position_to_node_id: Dict[Position, int] = dict()
    previous_time_point_position_to_node_id: Dict[Position, int] = dict()
    for time_point in experiment_positions.time_points():
        image_offset = experiment.images.offsets.of_time_point(time_point)

        # Iterate over all positions in this time point
        positions = list(experiment_positions.of_time_point(time_point))
        for position in positions:
            node_id = len(node_ids)

            # Add to node ids
            node_ids.append(node_id)
            current_time_point_position_to_node_id[position] = node_id

            # Add to edge ids
            for past_position in experiment_links.find_pasts(position):
                past_node_id = previous_time_point_position_to_node_id.get(past_position)
                if past_node_id is not None:
                    edge_ids.append((past_node_id, node_id))
                for data_name in edge_props.keys():
                    data_value = experiment_links.get_link_data(past_position, position, data_name)
                    edge_props[data_name].values.append(data_value)

            # Add to position properties (removing image offset, as GEFF doesn't support per-time-point offsets)
            node_x_array.values.append(position.x - image_offset.x)
            node_y_array.values.append(position.y - image_offset.y)
            node_z_array.values.append(position.z - image_offset.z)
            node_t_array.values.append(position.time_point_number())

        # Add to node properties (we can use a fast bulk method here)
        missing_data_names = set(position_data_names)
        for data_name, data_values in experiment_positions.create_time_point_dict(time_point, positions).items():
            node_props[data_name].values.extend(data_values)
            missing_data_names.discard(data_name)
        for missing_data_name in missing_data_names:
            # Don't forget to add None values for missing position data - otherwise the arrays will be misaligned
            node_props[missing_data_name].values.extend([None] * len(positions))

        # Move to next time point
        previous_time_point_position_to_node_id = current_time_point_position_to_node_id
        current_time_point_position_to_node_id = dict()

    # Convert global data to extra metadata
    write_arrays(
        geff_store=file_name,
        node_ids=numpy.array(node_ids, dtype=numpy.uint32),
        node_props={name: prop.to_geff_prop_dict() for name, prop in node_props.items()},
        edge_ids=numpy.array(edge_ids, dtype=numpy.uint32),
        edge_props={name: prop.to_geff_prop_dict() for name, prop in edge_props.items()},
        metadata=GeffMetadata(
            geff_version=geff.__version__,
            axes=[
                Axis(name="t", type="time", unit="frame"),
                Axis(name="z", type="space", unit="pixel"),
                Axis(name="y", type="space", unit="pixel"),
                Axis(name="x", type="space", unit="pixel")
            ],
            directed=True,
            node_props_metadata={},
            edge_props_metadata={},
            extra=experiment.global_data.get_all_data()
        ),
        overwrite=True
    )

