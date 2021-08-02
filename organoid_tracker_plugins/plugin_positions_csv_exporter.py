import json
import os
from typing import Dict, Any, List, Optional

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.linking_analysis import lineage_markers
from organoid_tracker.linking_analysis.cell_fate_finder import CellFateType


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export positions//CSV, as px coordinates...":
            lambda: _export_positions_px_as_csv(window),
        "File//Export-Export positions//CSV, as μm coordinates...":
            lambda: _export_positions_um_as_csv(window, metadata=False),
        "File//Export-Export positions//CSV, as μm coordinates with metadata...":
            lambda: _export_positions_um_as_csv(window, metadata=True),
    }


def _export_positions_px_as_csv(window: Window):
    experiment = window.get_experiment()
    if not experiment.positions.has_positions():
        raise UserError("No positions are found", "No annotated positions are found. Cannot export anything.")

    folder = dialog.prompt_save_file("Select a directory", [("Folder", "*")])
    if folder is None:
        return
    os.mkdir(folder)
    positions = experiment.positions

    file_prefix = experiment.name.get_save_name() + ".csv."
    for time_point in positions.time_points():
        offset = experiment.images.offsets.of_time_point(time_point)
        file_name = os.path.join(folder, file_prefix + str(time_point.time_point_number()))
        with open(file_name, "w") as file_handle:
            file_handle.write("x,y,z\n")
            for position in positions.of_time_point(time_point):
                position_px = position - offset
                file_handle.write(f"{position_px.x:.0f},{position_px.y:.0f},{position_px.z:.0f}\n")

    dialog.popup_message("Positions", "Exported all positions as CSV files.")


def _export_positions_um_as_csv(window: Window, *, metadata: bool):
    experiment = window.get_experiment()
    if not experiment.positions.has_positions():
        raise UserError("No positions are found", "No annotated positions are found. Cannot export anything.")

    folder = dialog.prompt_save_file("Select a directory", [("Folder", "*")])
    if folder is None:
        return
    os.mkdir(folder)
    if metadata:
        cell_types = list(window.get_gui_experiment().get_registered_markers(Position))
        window.get_scheduler().add_task(_AsyncExporter(experiment, cell_types, folder))
    else:
        _write_positions_to_csv(experiment, folder)
        _export_help_file(folder)
        dialog.popup_message("Positions", "Exported all positions, as well as a help file with instructions on how to"
                             " visualize the points in Paraview.")


def _export_help_file(folder: str, links: Optional[Links] = None):
    text = f"""
Instructions for importing in Paraview
======================================

1. Open Paraview and load the CSV files.
2. Press the green Apply button on the left. A table should appear.
3. Add a TableToPoints filter (Filters > Alphabetical). In the filter properties, set the coords (x, y, z) correctly.
4. Press the green Apply button on the left.
5. Make the filter visible by pressing the eye next to the filter in the Pipeline Browser.
6. Click somewhere in the 3D viewer to select the viewer.
7. In the filter properties, set the filter to be rendered using Points.
8. Set a Point Size of say 40 and set the checkmark next to "Render Points As Spheres"

You will now end up with a nice 3D view of the detected points. To save a movie, use File > Export animation.
"""
    if links is not None:
        text += f"""

How to color the lineages correctly
===================================
This assumes you want to color the lineage trees in the same way as OrganoidTracker does.
1. Select the TableToPoints filter and color by original_track_id (to color all cells)
   or lineage_id (to color only cells that have divided or will divide during the time lapse).
2. On the right of the screen you see a color panel. Make sure "Interpret Values as Categories" is OFF.
3. Click "Choose Preset" (small square button with an icon, on the right of the "Mapping Data" color graph).
4. In the dropdown-menu on the top right, change "default" into "All". 
5. Import the lineage_colormap.json file from this folder, press Apply and close the popup. The colors of the spheres
   should change.
6. Click "Rescale to custom range" (similar button to "Choose Preset") and set the scale from -1 to {links.get_highest_track_id()}.
7. Make sure "Color Discretization" (near the bottom of color panel) is off.

How to color the cell types correctly
=====================================
This assumes you want to color the cell types in the same way as OrganoidTracker does.
1. Select the TableToPoints filter and color by the cell type id.
2. On the right of the screen you see a color panel. Make sure "Interpret Values as Categories" is ON.
3. Click "Choose Preset" (small square button with an icon, on the right of the "Mapping Data" color graph).
4. Press the gear icon near the top right of the screen of the popup to show the advanced options.
5. Make sure that on the top right of the screen, the checkmark next to "Annotations" (below "Options to load") is checked.
5. Import the cell_type_colormap.json file from this folder, press Apply and close the popup. The colors of the spheres
   should change.
"""
    file_name = os.path.join(folder, "How to import to Paraview.txt")
    with open(file_name, "w") as file_handle:
        file_handle.write(text)


def _write_positions_to_csv(experiment: Experiment, folder: str):
    resolution = experiment.images.resolution()
    positions = experiment.positions

    file_prefix = experiment.name.get_save_name() + ".csv."
    for time_point in positions.time_points():
        file_name = os.path.join(folder, file_prefix + str(time_point.time_point_number()))
        with open(file_name, "w") as file_handle:
            file_handle.write("x,y,z\n")
            for position in positions.of_time_point(time_point):
                vector = position.to_vector_um(resolution)
                file_handle.write(f"{vector.x},{vector.y},{vector.z}\n")


class _CellTypesToId:
    """Gives a sequential id to every cell type"""
    _save_name_to_id: Dict[str, int]

    def __init__(self):
        self._save_name_to_id = dict()

    def get_or_add_id(self, save_name: Optional[str]) -> int:
        """Gets the id of the given cell type. Creates a new id if this type doesn't have an id yet. Returns -1 if
        (and only if) the input value is None."""
        if save_name is None:
            return -1
        id = self._save_name_to_id.get(save_name)
        if id is None:
            # Create new id
            id = len(self._save_name_to_id)
            self._save_name_to_id[save_name] = id
        return id

    def get_id(self, save_name: str) -> Optional[int]:
        """Gets the id of the cell type, if registered."""
        return self._save_name_to_id.get(save_name)


def _export_cell_types_file(folder: str, cell_types: List[Marker], cell_type_ids: _CellTypesToId):
    """Writes all known cell types and their ids to a file."""
    if len(cell_types) == 0:
        return

    annotations_list = list()
    indexed_colors_list = list()
    for cell_type in cell_types:
        id = cell_type_ids.get_id(cell_type.save_name)
        if id is not None:
            annotations_list.append(str(id))
            annotations_list.append(str(cell_type.display_name))
            indexed_colors_list.append(cell_type.mpl_color[0])
            indexed_colors_list.append(cell_type.mpl_color[1])
            indexed_colors_list.append(cell_type.mpl_color[2])

    # Add NaN for unknown cell types
    annotations_list.append("-1")
    annotations_list.append("Unknown type")
    indexed_colors_list.append(1)
    indexed_colors_list.append(1)
    indexed_colors_list.append(1)

    data = [
        {
            "Annotations": annotations_list,
            "IndexedColors": indexed_colors_list,
            "Name": "Cell types (uses annotations)"
        }
    ]

    # Write data
    file_name = os.path.join(folder, "cell_type_colormap.json")
    with open(file_name, "w") as file_handle:
        json.dump(data, file_handle)


class _AsyncExporter(Task):
    _positions: PositionCollection
    _links: Links
    _position_data: PositionData
    _resolution: ImageResolution
    _folder: str
    _save_name: str
    _registered_cell_types: List[Marker]
    _cell_types_to_id: _CellTypesToId
    _division_lookahead_time_points: int

    def __init__(self, experiment: Experiment, registered_cell_types: List[Marker], folder: str):
        self._positions = experiment.positions.copy()
        self._links = experiment.links.copy()
        self._position_data = experiment.position_data.copy()
        self._resolution = experiment.images.resolution()
        self._folder = folder
        self._save_name = experiment.name.get_save_name()
        self._registered_cell_types = registered_cell_types
        self._cell_types_to_id = _CellTypesToId()
        self._division_lookahead_time_points = experiment.division_lookahead_time_points

    def compute(self) -> Any:
        self._links.sort_tracks_by_x()

        _write_positions_and_metadata_to_csv(self._positions, self._position_data, self._links, self._resolution,
                                             self._cell_types_to_id, self._division_lookahead_time_points,
                                             self._folder, self._save_name)
        _export_help_file(self._folder, self._links)
        _export_cell_types_file(self._folder, self._registered_cell_types, self._cell_types_to_id)
        _export_colormap_file(self._folder, self._links)
        return "done"  # We're not using the result

    def on_finished(self, result: Any):
        dialog.popup_message("Positions", "Exported all positions, as well as a help file with instructions on how to"
                                          " visualize the points in Paraview.")


def _write_positions_and_metadata_to_csv(positions: PositionCollection, position_data: PositionData, links: Links,
                                         resolution: ImageResolution, cell_types_to_id: _CellTypesToId,
                                         division_lookahead_time_points: int, folder: str, save_name: str):
    from organoid_tracker.linking import cell_division_finder
    from organoid_tracker.position_analysis import cell_density_calculator
    from organoid_tracker.linking_analysis import lineage_id_creator, cell_division_counter, cell_nearby_death_counter,\
        cell_fate_finder, cell_compartment_finder

    deaths_nearby_tracks = cell_nearby_death_counter.NearbyDeaths(links, position_data, resolution)
    first_time_point_number = positions.first_time_point_number()

    file_prefix = save_name + ".csv."
    for time_point in positions.time_points():
        file_name = os.path.join(folder, file_prefix + str(time_point.time_point_number()))
        with open(file_name, "w") as file_handle:
            file_handle.write("x,y,z,density_mm1,times_divided,times_neighbor_died,cell_in_dividing_compartment,"
                              "cell_type_id,hours_until_division,hours_until_dead,hours_since_division,lineage_id,"
                              "original_track_id\n")
            positions_of_time_point = positions.of_time_point(time_point)
            for position in positions_of_time_point:
                lineage_id = lineage_id_creator.get_lineage_id(links, position)
                original_track_id = lineage_id_creator.get_original_track_id(links, position)
                cell_type_id = cell_types_to_id.get_or_add_id(
                    position_markers.get_position_type(position_data, position))
                density = cell_density_calculator.get_density_mm1(positions_of_time_point, position, resolution)
                times_divided = cell_division_counter.find_times_divided(links, position, first_time_point_number)
                times_neighbor_died = deaths_nearby_tracks.count_nearby_deaths_in_past(links, position)
                cell_compartment_id = cell_compartment_finder.find_compartment_ext(positions, links, resolution,
                                             division_lookahead_time_points, position).value
                if cell_compartment_id == cell_compartment_finder.CellCompartment.UNKNOWN.value:
                    cell_compartment_id = None  # Set to none if unknown
                cell_fate = cell_fate_finder.get_fate_ext(links, position_data, division_lookahead_time_points, position)
                hours_until_division = cell_fate.time_points_remaining * resolution.time_point_interval_h \
                        if cell_fate.type == CellFateType.WILL_DIVIDE else -1
                hours_until_dead = cell_fate.time_points_remaining * resolution.time_point_interval_h \
                        if cell_fate.type in cell_fate_finder.WILL_DIE_OR_SHED else -1
                previous_division = cell_division_finder.get_previous_division(links, position)
                hours_since_division = (time_point.time_point_number() - previous_division.mother.time_point_number())\
                                       * resolution.time_point_interval_h if previous_division is not None else None
                if cell_fate.type == CellFateType.UNKNOWN:  # If unknown, set to None
                    hours_until_dead = None
                    hours_until_division = None

                vector = position.to_vector_um(resolution)
                file_handle.write(f"{vector.x},{vector.y},{vector.z},{density},{_str(times_divided)},"
                                  f"{times_neighbor_died},{_str(cell_compartment_id)},{_str(cell_type_id)},"
                                  f"{_str(hours_until_division)},{_str(hours_until_dead)},{_str(hours_since_division)},"
                                  f"{lineage_id},{original_track_id}\n")


def _export_colormap_file(folder: str, links: Links):
    """Writes the given Matplotlib colormap as a Paraview JSON colormap."""
    # Create colormap
    rgb_points = [-1, 1, 1, 1]  # Define track -1 as white
    max_track_id = links.get_highest_track_id()
    for i, track in links.find_all_tracks_and_ids():
        lineage_color = lineage_markers.get_color(links, track)

        # If color is fully black, change it to white
        color_tuple = lineage_color.to_rgb_floats() if not lineage_color.is_black() else (1, 1, 1)

        rgb_points.append(i)
        rgb_points.append(color_tuple[0])
        rgb_points.append(color_tuple[1])
        rgb_points.append(color_tuple[2])

    # Create data
    data = [
        {
            "Colorspace": "RGB",
            "Name": f"Lineage colors (-1 to {max_track_id})",
            "RGBPoints": rgb_points
        }
    ]

    # Write data
    file_name = os.path.join(folder, "lineage_colormap.json")
    with open(file_name, "w") as file_handle:
        json.dump(data, file_handle)


def _str(value: Optional[float]) -> str:
    """Converts None to "NaN"."""
    if value is None:
        return "NaN"
    return str(value)
