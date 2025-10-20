# This file was based on https://github.com/live-image-tracking-tools/geff ,
# and adapted to load into our data structures (instead of a NetworkX graph or whatever).
#
# MIT License
#
# Copyright (c) 2024 Funke lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Literal, NamedTuple
from typing import TypedDict, Any, NotRequired

import numpy
import zarr
from numpy.typing import NDArray
from zarr.storage import StoreLike

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.core.resolution import ImageResolution

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

class _GeffMetadata(TypedDict):
    """Metadata for a GEFF graph, following https://liveimagetrackingtools.org/geff/latest/specification/ ."""
    geff_version: str
    directed: bool
    axes: list[dict[str, Any]]
    node_props_metadata: dict[str, dict[str, Any]]
    edge_props_metadata: dict[str, dict[str, Any]]
    sphere: str | None
    ellipsoid: str | None
    track_node_props: dict[str, Any] | None
    related_objects: list[dict[str, Any]] | None
    display_hints: dict[str, Any] | None
    extra: dict[str, Any]


class _PropDictNpArray(NamedTuple):
    """A prop dictionary which has the keys "values" and optionally "missing", stored as numpy arrays."""
    values: NDArray[Any]
    missing: NDArray[numpy.bool_] | None


class _ZarrPropDict(TypedDict):
    """A prop dictionary which has the keys "values" and optionally "missing" and "data", stored as zarr arrays."""
    values: zarr.Array
    missing: NotRequired[zarr.Array]
    data: NotRequired[zarr.Array]


class _InMemoryGeff(TypedDict):
    """Geff data loaded into memory as numpy arrays."""

    metadata: _GeffMetadata
    node_ids: NDArray[Any]
    edge_ids: NDArray[Any]
    node_props: dict[str, _PropDictNpArray]
    edge_props: dict[str, _PropDictNpArray]


def _load_prop_to_memory(zarr_prop: _ZarrPropDict, prop_metadata: dict[str, Any]) -> _PropDictNpArray:
    """Load a zarr property dictionary into memory, including deserialization."""
    dtype = numpy.dtype(prop_metadata["dtype"])
    values_dtype = numpy.uint64 if prop_metadata["varlength"] else dtype
    values = numpy.array(zarr_prop["values"], dtype=values_dtype)
    if "missing" in zarr_prop:
        missing = numpy.array(zarr_prop["missing"], dtype=bool)
    else:
        missing = None
        
    if "data" in zarr_prop:
        data = numpy.array(zarr_prop["data"], dtype=dtype)
    else:
        data = None

    if "varlength" in prop_metadata and prop_metadata["varlength"]:
        if data is None:
            raise ValueError(
                f"Property {prop_metadata['identifier']} metadata is varlength but no "
                "serialized data was found in GEFF zarr"
            )
        return _deserialize_vlen_property_data(values, missing, data)
    else:
        return _PropDictNpArray(values=values, missing=missing)


def _read_prop(group: zarr.Group, name: str, prop_type: Literal["node", "edge"]) -> _ZarrPropDict:
    """Read a single property into a zarr property dictionary."""
    group_path = f"nodes/props/{name}" if prop_type == "node" else f"edges/props/{name}"
    prop_group = zarr.open_group(group.store, path=group_path, mode="r")
    values: zarr.Array = prop_group.get("values")
    prop_dict: _ZarrPropDict = {"values": values}
    if "missing" in prop_group.keys():
        missing = prop_group.get("missing")
        prop_dict["missing"] = missing

    if "data" in prop_group.keys():
        prop_dict["data"] = prop_group.get("data")
    return prop_dict


def _read_geff(source: StoreLike) -> _InMemoryGeff:
    """Parses a GEFF zarr store into an in-memory representation."""
    group = zarr.open_group(source, mode="r")
    metadata: _GeffMetadata = group.attrs["geff"]  # type: ignore
    nodes = zarr.open_array(source, path="nodes/ids", mode="r")
    edges = zarr.open_array(source, path="edges/ids", mode="r")
    node_props: dict[str, _ZarrPropDict] = {}
    edge_props: dict[str, _ZarrPropDict] = {}

    # get node properties
    nodes_group = group.get("nodes")
    if "props" in nodes_group.keys():
        node_props_group = zarr.open_group(group.store, path="nodes/props", mode="r")
        node_prop_names: list[str] = [*node_props_group.group_keys()]
    else:
        node_prop_names = []
    for name in node_prop_names:
        node_props[name] = _read_prop(group, name, "node")

    # get edge properties
    edges_group = group.get("edges")
    if "props" in edges_group.keys():
        edge_props_group = zarr.open_group(group.store, path="edges/props", mode="r")
        edge_prop_names: list[str] = [*edge_props_group.group_keys()]
    else:
        edge_prop_names = []
    for name in edge_prop_names:
        edge_props[name] = _read_prop(group, name, "edge")

    nodes = numpy.array(nodes)
    node_props_memory: dict[str, _PropDictNpArray] = {}
    for name, props in node_props.items():
        prop_metadata = metadata["node_props_metadata"][name]
        node_props_memory[name] = _load_prop_to_memory(props, prop_metadata)

    # remove edges if any of its nodes has been masked
    edges = numpy.array(edges[:])
    edge_props_memory: dict[str, _PropDictNpArray] = {}
    for name, props in edge_props.items():
        prop_metadata = metadata["edge_props_metadata"][name]
        edge_props_memory[name] = _load_prop_to_memory(props, prop_metadata)


    # Remove any metadata for properties that are not present
    for property_name in list(metadata["node_props_metadata"].keys()):
        if property_name not in node_props.keys():
            del metadata["node_props_metadata"][property_name]
    for property_name in list(metadata["edge_props_metadata"].keys()):
        if property_name not in edge_props.keys():
            del metadata["edge_props_metadata"][property_name]

    return {
        "metadata": metadata,
        "node_ids": nodes,
        "node_props": node_props_memory,
        "edge_ids": edges,
        "edge_props": edge_props_memory,
    }


def _deserialize_vlen_property_data(values: NDArray, missing: NDArray[numpy.bool_] | None,
                                    data: NDArray) -> _PropDictNpArray:
    """Deserialize a variable-length property from the values and data arrays."""
    decoded_values = numpy.empty(shape=(len(values),), dtype=numpy.object_)
    for i in range(values.shape[0]):
        decoded_values[i] = _deserialize_vlen_value(values, data, i)
    return _PropDictNpArray(values=decoded_values, missing=missing)


def _deserialize_vlen_value(values: NDArray, data: NDArray, index: int) -> NDArray:
    """Deserialize one variable-length array value from the data and values arrays."""
    if index < 0 or index >= values.shape[0]:
        raise IndexError(f"Index {index} out of bounds for property data.")
    offset = values[index][0]
    shape = values[index][1:]
    return data[offset: offset + numpy.prod(shape)].reshape(shape)


def load_data_file(file_name: str, min_time_point: int = -9999999, max_time_point: int = 9999999, experiment: Experiment = None):
    experiment = experiment if experiment is not None else Experiment()

    try:
        geff_index = file_name.lower().rindex(".geff")
    except ValueError:
        raise ValueError("File name does not contain .geff extension")
    geff_source = file_name[:geff_index + 5]

    in_memory_geff = _read_geff(geff_source)

    # Read in the metadata names
    node_prop_names = list()
    for node_prop_info in in_memory_geff["metadata"]["node_props_metadata"].values():
        # Name, description and unit not handled yet
        identifier = node_prop_info["identifier"]
        if identifier in ["t", "x", "y", "z"]:
            continue  # Skip position properties
        node_prop_names.append(identifier)
    edge_prop_names = list()
    for edge_prop_info in in_memory_geff["metadata"]["edge_props_metadata"].values():
        identifier = edge_prop_info["identifier"]
        edge_prop_names.append(identifier)

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
            prop_values = prop_array.values
            prop_missing = prop_array.missing
            if prop_missing is not None and prop_missing[i]:
                continue  # Missing value
            experiment_links.set_link_data(source_position, target_position, edge_prop_name, prop_values[i].tolist())

    # Read in any extra metadata
    for key, value in in_memory_geff["metadata"]["extra"].items():
        if isinstance(value, numpy.generic):
            value = value.tolist()  # Convert numpy scalar to native Python type

        experiment.global_data.set_data(key, value)

    return experiment


def _read_positions(experiment: Experiment, in_memory_geff: _InMemoryGeff, min_time_point: int, max_time_point: int,
                    node_prop_names: list[str]) -> list[Position | None]:
    """Read the positions and position metadata from the in-memory GEFF data and adds them to the experiment.
    Returns a list of positions by node id."""

    # Read in the scale factors
    geff_axes = in_memory_geff["metadata"]["axes"]
    time_scale_factor, z_scale_factor, y_scale_factor, x_scale_factor = _read_scale_factors_tzyx(experiment, geff_axes)

    node_ids = in_memory_geff["node_ids"]
    node_props = in_memory_geff["node_props"]
    time_values = node_props["t"].values
    x_values = node_props["x"].values
    y_values = node_props["y"].values
    z_values = node_props["z"].values

    positions_by_node_id = []
    experiment_positions = experiment.positions
    for i in range(len(node_ids)):
        node_id = node_ids[i]

        time_point_number = int(time_values[i])
        if time_point_number < min_time_point or time_point_number > max_time_point:
            # Skip positions outside of the requested time point range
            continue

        position = Position(float(x_values[i] * x_scale_factor),
                            float(y_values[i] * y_scale_factor),
                            float(z_values[i] * z_scale_factor),
                            time_point_number=int(time_values[i] * time_scale_factor))

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
            prop_values = prop_array.values
            prop_missing = prop_array.missing
            if prop_missing is not None and prop_missing[i]:
                continue  # Missing value
            experiment_positions.set_position_data(position, node_prop_name, prop_values[i].tolist())
    return positions_by_node_id


def _read_scale_factors_tzyx(experiment: Experiment, geff_axes: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """Read the scale factors for time, z, y, x axes from the GEFF axes. Raises an UserError if unsupported units are found."""
    x_scale_factor = 1
    y_scale_factor = 1
    z_scale_factor = 1
    time_scale_factor = 1
    for ax in geff_axes:
        ax_name = ax["name"]

        # Scale and offset are currently not supported
        if "scale" in ax and ax["scale"] is not None:
            raise UserError("Unsupported file", "Scaled axes are not supported in our GEFF loader")
        if "offset" in ax and ax["offset"] is not None:
            raise UserError("Unsupported file", "Offsetting axes is not supported in our GEFF loader")

        if ax["type"] == "time":
            if ax["unit"] is not None and ax["unit"] != "frame":
                resolution = _get_resolution(experiment)
                if ax["unit"] not in _MULTIPLICATION_FACTOR_TO_MINUTES:
                    raise UserError("Unsupported file",
                                    f"The unit '{ax['unit']}' for the time axis '{ax_name}' is not supported in our GEFF loader")
                time_scale_factor = _MULTIPLICATION_FACTOR_TO_MINUTES[ax["unit"]] / resolution.time_point_interval_m
        elif ax["type"] == "space":
            if ax["unit"] is not None and ax["unit"] != "pixel":
                resolution = _get_resolution(experiment)
                if ax["unit"] not in _MULTIPLICATION_FACTOR_TO_MICROMETERS:
                    raise UserError("Unsupported file",
                                    f"The unit '{ax['unit']}' for the {ax_name}-axis is not supported in our GEFF loader")
                scale_to_micrometers = _MULTIPLICATION_FACTOR_TO_MICROMETERS[ax["unit"]]
                if ax_name == "x":
                    x_scale_factor = scale_to_micrometers / resolution.pixel_size_x_um
                elif ax_name == "y":
                    y_scale_factor = scale_to_micrometers / resolution.pixel_size_y_um
                elif ax_name == "z":
                    z_scale_factor = scale_to_micrometers / resolution.pixel_size_z_um
                else:
                    raise UserError("Unsupported file",
                                    f"Spatial axis name '{ax_name}' is not supported in our GEFF loader")
        elif ax["type"] == "channel":
            continue
        else:
            raise NotImplementedError(f"Axis type {ax['type']} is not supported in our GEFF loader")
    return time_scale_factor, z_scale_factor, y_scale_factor, x_scale_factor


def _get_resolution(experiment: Experiment) -> ImageResolution:
    try:
        resolution = experiment.images.resolution()
    except UserError:
        raise UserError("Spatial resolution error",
                        "GEFF file has positions in spatial units, and we need to convert them to pixel positions."
                        " However, we're missing the image resolution, so we cannot do this."
                        "\n\nUsually the resolution is set when loading the images, and otherwise, use 'Edit > Set image resolution...' to set it manually.")
    return resolution


if __name__ == "__main__":
    load_data_file(r"E:\Scratch\GEFF-test\Fluo-N3DH-CE\01_GT.geff")