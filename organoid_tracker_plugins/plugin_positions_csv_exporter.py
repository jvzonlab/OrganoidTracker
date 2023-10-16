import csv
import json
import os
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union, Iterable

import numpy

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.links import Links
from organoid_tracker.core.marker import Marker
from organoid_tracker.core.position import Position
from organoid_tracker.core.position_collection import PositionCollection
from organoid_tracker.core.position_data import PositionData
from organoid_tracker.core.resolution import ImageResolution
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.linking_analysis.cell_nearby_death_counter import NearbyDeaths
from organoid_tracker.position_analysis import position_markers
from organoid_tracker.linking_analysis import lineage_markers
from organoid_tracker.linking_analysis.cell_fate_finder import CellFateType


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export positions//CSV, as px coordinates...":
            lambda: _export_positions_um_as_csv(window, metadata=False, use_micrometer=False),
        "File//Export-Export positions//CSV, as px coordinates with metadata...":
            lambda: _export_positions_um_as_csv(window, metadata=True, use_micrometer=False),
        "File//Export-Export positions//CSV, as μm coordinates...":
            lambda: _export_positions_um_as_csv(window, metadata=False, use_micrometer=True),
        "File//Export-Export positions//CSV, as μm coordinates with metadata...":
            lambda: _export_positions_um_as_csv(window, metadata=True, use_micrometer=True),
    }


class _BuiltInDataNames(Enum):
    lineage_id = auto()
    ancestor_track_id = auto()
    track_id = auto()
    cell_type_id = auto()
    density_mm1 = auto()
    times_divided = auto()
    times_neighbor_died = auto()
    cell_compartment_id = auto()
    hours_until_division = auto()
    hours_until_dead = auto()
    hours_since_division = auto()

def _get_data_names(window: Window) -> List[str]:
    answers = set()

    # Add metadata we can calculate on the fly
    for built_in_name in _BuiltInDataNames:
        answers.add(built_in_name.name)

    # Find metadata in position_data that we can store in CSV files
    for experiment in window.get_active_experiments():
        for data_name, data_type in experiment.position_data.get_data_names_and_types().items():
            if data_type in {float, bool, int, str, list}:
                answers.add(data_name)

    answers = list(answers)
    answers.sort()
    return answers


def _export_positions_um_as_csv(window: Window, *, metadata: bool, use_micrometer: bool):
    experiments = list(window.get_active_experiments())
    for experiment in experiments:
        if not experiment.positions.has_positions():
            raise UserError("No positions are found", f"No annotated positions are found in experiment"
                                                      f" \"{experiment.name}\". Cannot export anything.")

    folder = dialog.prompt_save_file("Select a directory", [("Folder", "*")])
    if folder is None:
        return
    if metadata:
        available_data_names = _get_data_names(window)
        answer = option_choose_dialog.prompt_list_multiple("Exporting positions", "Metadata names", "Metadata to export:",
                                                  available_data_names)
        if answer is None:
            return
        data_names = [available_data_names[i] for i in answer]
        cell_types = list(window.registry.get_registered_markers(Position))
        window.get_scheduler().add_task(_AsyncExporter(experiments, cell_types, data_names, folder,
                                                       use_micrometer=use_micrometer))
    else:
        for i, experiment in enumerate(experiments):
            positions_folder = folder if len(experiments) == 1 else os.path.join(folder, str(i + 1) + ". " + experiment.name.get_save_name())
            os.makedirs(positions_folder, exist_ok=True)
            _write_positions_to_csv(experiment, positions_folder, use_micrometer=use_micrometer)
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
1. Select the TableToPoints filter and color by ancestor_track_id (to color all cells)
   or lineage_id (to color only cells that have divided or will divide during the time lapse).
   (If "ancestor_track_id" or "lineage_id" is missing from your table, then you need to re-export. This time,
   select one or both of those values during exporting.)
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
1. Select the TableToPoints filter and color by the cell type id. If the cell type id value is missing, then
   you'll need to re-export. This time, make sure to select "cell_type_id" as one of the values that you export.)
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


def _write_positions_to_csv(experiment: Experiment, folder: str, *, use_micrometer: bool):
    resolution = experiment.images.resolution(allow_incomplete=not use_micrometer)
    positions = experiment.positions

    file_prefix = experiment.name.get_save_name() + ".csv."
    for time_point in positions.time_points():
        file_name = os.path.join(folder, file_prefix + str(time_point.time_point_number()))
        with open(file_name, "w") as file_handle:
            file_handle.write("x,y,z\n")
            for position in positions.of_time_point(time_point):
                vector = position.to_vector_um(resolution) if use_micrometer else position
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
    _experiments: List[Experiment]
    _use_micrometer: bool
    _folder: str
    _registered_cell_types: List[Marker]
    _cell_types_to_id: _CellTypesToId
    _data_names: List[str]

    def __init__(self, experiments: List[Experiment], registered_cell_types: List[Marker], data_names: List[str],
                 folder: str, *, use_micrometer: bool):
        self._data_names = data_names
        self._folder = folder
        self._registered_cell_types = registered_cell_types
        self._cell_types_to_id = _CellTypesToId()
        self._use_micrometer = use_micrometer
        self._experiments = [self._copy(experiment) for experiment in experiments]

    def _copy(self, experiment: Experiment):
        copy = experiment.copy_selected(positions=True, links=True, position_data=True)
        copy.name.set_name(experiment.name.get_name(), is_automatic=experiment.name.is_automatic())
        copy.images.set_resolution(experiment.images.resolution(allow_incomplete=not self._use_micrometer))
        return copy

    def compute(self) -> Any:
        for i, experiment in enumerate(self._experiments):
            folder = self._folder if len(self._experiments) == 1\
                else os.path.join(self._folder, str(i + 1) + ". " + experiment.name.get_save_name())
            os.makedirs(folder, exist_ok=True)
            experiment.links.sort_tracks_by_x()

            _write_positions_and_metadata_to_csv(self._data_names, experiment.positions, experiment.position_data,
                   experiment.links, experiment.images.resolution(allow_incomplete=True), self._cell_types_to_id,
                   experiment.division_lookahead_time_points, folder, experiment.name.get_save_name(),
                                                 use_micrometer=self._use_micrometer)
            _export_help_file(folder, experiment.links)
            _export_cell_types_file(folder, self._registered_cell_types, self._cell_types_to_id)
            _export_colormap_file(folder, experiment.links)
        return "done"  # We're not using the result

    def on_finished(self, result: Any):
        dialog.popup_message("Positions", "Exported all positions, as well as a help file with instructions on how to"
                                          " visualize the points in Paraview.")


def _get_metadata(position: Position, data_name: str, positions: PositionCollection, position_data: PositionData,
                  links: Links, resolution: ImageResolution, cell_types_to_id: _CellTypesToId,
                  deaths_nearby_tracks: NearbyDeaths,
                  division_lookahead_time_points: int) -> Union[str, float]:
    if data_name == "lineage_id":
        from organoid_tracker.linking_analysis import lineage_id_creator
        lineage_id = lineage_id_creator.get_lineage_id(links, position)
        return lineage_id

    if data_name == "ancestor_track_id":
        from organoid_tracker.linking_analysis import lineage_id_creator
        ancestor_track_id = lineage_id_creator.get_original_track_id(links, position)
        return ancestor_track_id
    
    if data_name == "track_id":
        track = links.get_track(position)
        if track is None:
            return -1
        return links.get_track_id(track)

    if data_name == "cell_type_id":
        cell_type_id = cell_types_to_id.get_or_add_id(
            position_markers.get_position_type(position_data, position))
        return cell_type_id

    if data_name == "density_mm1":
        from organoid_tracker.position_analysis import cell_density_calculator
        if not resolution.is_incomplete(require_time_resolution=False):
            raise UserError("No resolution set", "No resolution was set. Cannot calculate the density.")
        positions_of_time_point = positions.of_time_point(position.time_point())
        density = cell_density_calculator.get_density_mm1(positions_of_time_point, position, resolution)
        return density

    if data_name == "times_divided":
        from organoid_tracker.linking_analysis import cell_division_counter
        first_time_point_number = positions.first_time_point_number()
        times_divided = cell_division_counter.find_times_divided(links, position, first_time_point_number)
        return times_divided

    if data_name == "times_neighbor_died":
        times_neighbor_died = deaths_nearby_tracks.count_nearby_deaths_in_past(links, position)
        return times_neighbor_died

    if data_name == "cell_compartment_id":
        from organoid_tracker.linking_analysis import cell_compartment_finder
        if not resolution.is_incomplete():
            raise UserError("No resolution set", "No resolution was set. Cannot calculate the density.")
        cell_compartment_id = cell_compartment_finder.find_compartment_ext(positions, links, resolution,
                                                                           division_lookahead_time_points,
                                                                           position).value
        if cell_compartment_id == cell_compartment_finder.CellCompartment.UNKNOWN.value:
            cell_compartment_id = None  # Set to none if unknown
        return cell_compartment_id

    if data_name == "hours_until_division" or data_name == "hours_until_dead" or data_name == "hours_since_division":
        from organoid_tracker.linking import cell_division_finder
        from organoid_tracker.linking_analysis import cell_fate_finder
        if not resolution.is_incomplete():
            raise UserError("No resolution set", "No resolution was set. Cannot calculate timings.")

        cell_fate = cell_fate_finder.get_fate_ext(links, position_data, division_lookahead_time_points,
                                                  position)
        hours_until_division = cell_fate.time_points_remaining * resolution.time_point_interval_h \
            if cell_fate.type == CellFateType.WILL_DIVIDE else -1
        hours_until_dead = cell_fate.time_points_remaining * resolution.time_point_interval_h \
            if cell_fate.type in cell_fate_finder.WILL_DIE_OR_SHED else -1
        previous_division = cell_division_finder.get_previous_division(links, position)
        hours_since_division = (position.time_point_number() - previous_division.mother.time_point_number()) \
                               * resolution.time_point_interval_h if previous_division is not None else None
        if cell_fate.type == CellFateType.UNKNOWN:  # If unknown, set to None
            hours_until_dead = None
            hours_until_division = None

        if data_name == "hours_until_division":
            return hours_until_division
        if data_name == "hours_since_division":
            return hours_since_division
        return hours_until_dead

    value = position_data.get_position_data(position, data_name)
    if isinstance(value, list):
        return json.dumps(value)[1:-1]  # Easy way of serializing a list. [2, 3, 4] will become "2, 3, 4"
    return value


def _write_positions_and_metadata_to_csv(data_names: List[str], positions: PositionCollection,
                                         position_data: PositionData, links: Links,
                                         resolution: ImageResolution, cell_types_to_id: _CellTypesToId,
                                         division_lookahead_time_points: int, folder: str, save_name: str,
                                         *, use_micrometer: bool):
    from organoid_tracker.linking_analysis import cell_nearby_death_counter

    deaths_nearby_tracks = cell_nearby_death_counter.NearbyDeaths(links, position_data, resolution)

    file_prefix = save_name + ".csv."
    for time_point in positions.time_points():
        file_name = os.path.join(folder, file_prefix + str(time_point.time_point_number()))
        with open(file_name, "w", newline='') as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(["x", "y", "z"] + data_names)
            positions_of_time_point = positions.of_time_point(time_point)
            for position in positions_of_time_point:
                vector = position.to_vector_um(resolution) if use_micrometer else position
                data_row = [vector.x, vector.y, vector.z]
                for data_name in data_names:
                    value = _get_metadata(position, data_name, positions, position_data, links, resolution,
                                          cell_types_to_id, deaths_nearby_tracks, division_lookahead_time_points)
                    if value is None:
                        value = "NaN"
                    data_row.append(value)
                writer.writerow(data_row)


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
