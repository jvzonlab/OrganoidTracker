import json
import os
from typing import Dict, Any, List

import matplotlib.colors

from ai_track.core import UserError
from ai_track.core.experiment import Experiment
from ai_track.core.links import Links
from ai_track.core.marker import Marker
from ai_track.core.position import Position
from ai_track.core.position_collection import PositionCollection
from ai_track.core.resolution import ImageResolution
from ai_track.gui import dialog
from ai_track.gui.threading import Task
from ai_track.gui.window import Window
from ai_track.linking_analysis import lineage_id_creator


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Export-Export positions//CSV, as μm coordinates...": lambda: _export_positions_as_csv(window, metadata=False),
        "File//Export-Export positions//CSV, as μm coordinates with metadata...": lambda: _export_positions_as_csv(window, metadata=True),
    }


def _export_positions_as_csv(window: Window, *, metadata: bool):
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


def _export_help_file(folder: str, links: Links):
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

How to color the lineages correctly
===================================
This assumes you want to color the lineage trees in the same way as AI_track does.
1. Select the TableToPoints filter and color by lineage id.
2. On the right of the screen you see a color panel. Make sure "Interpret Values as Categories" is off.
3. Click "Choose Preset" (small square button with an icon, on the right of the "Mapping Data" color graph).
4. Press the gear icon near the top right of the screen of the popup. 
5. Import the lineage_colormap.json file from this folder, press Apply and close the popup. The colors of the spheres
   should change.
6. Click "Rescale to custom range" (similar button to "Choose Preset") and set the scale from -1 to {links.get_highest_track_id()}.
7. Make sure "Color Discretization" (near the bottom of color panel) is off.
"""
    file_name = os.path.join(folder, "How to import to Paraview.txt")
    with open(file_name, "w") as file_handle:
        file_handle.write(text)


def _export_cell_types_file(folder: str, cell_types: List[Marker]):
    """Writes all known cell types and their ids to a file."""
    if len(cell_types) == 0:
        return
    file_name = os.path.join(folder, "Known cell types.txt")
    with open(file_name, "w") as file_handle:
        file_handle.write("Known cell types:\n")
        file_handle.write("=================\n")
        for cell_type in cell_types:
            file_handle.write("* " + cell_type.display_name + " = " + str(hash(cell_type.save_name)))


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


class _AsyncExporter(Task):
    _positions: PositionCollection
    _links: Links
    _resolution: ImageResolution
    _folder: str
    _save_name: str
    _cell_types: List[Marker]

    def __init__(self, experiment: Experiment, cell_types: List[Marker], folder: str):
        self._positions = experiment.positions.copy()
        self._links = experiment.links.copy()
        self._resolution = experiment.images.resolution()
        self._folder = folder
        self._save_name = experiment.name.get_save_name()
        self._cell_types = cell_types

    def compute(self) -> Any:
        self._links.sort_tracks_by_x()

        _write_positions_and_metadata_to_csv(self._positions, self._links, self._resolution, self._folder, self._save_name)
        _export_help_file(self._folder, self._links)
        _export_cell_types_file(self._folder, self._cell_types)
        _export_colormap_file(self._folder, self._links)
        return "done"  # We're not using the result

    def on_finished(self, result: Any):
        dialog.popup_message("Positions", "Exported all positions, as well as a help file with instructions on how to"
                                          " visualize the points in Paraview.")


def _write_positions_and_metadata_to_csv(positions: PositionCollection, links: Links, resolution: ImageResolution, folder: str, save_name: str):
    from ai_track.linking_analysis import lineage_id_creator
    from ai_track.position_analysis import cell_density_calculator
    from ai_track.linking_analysis import linking_markers, cell_division_counter, cell_nearby_death_counter

    deaths_nearby_tracks = cell_nearby_death_counter.NearbyDeaths(links, resolution)
    first_time_point_number = positions.first_time_point_number()

    file_prefix = save_name + ".csv."
    for time_point in positions.time_points():
        file_name = os.path.join(folder, file_prefix + str(time_point.time_point_number()))
        with open(file_name, "w") as file_handle:
            file_handle.write("x,y,z,density_mm1,times_divided,times_neighbor_died,cell_type_id,"
                              "lineage_id,original_track_id\n")
            positions_of_time_point = positions.of_time_point(time_point)
            for position in positions_of_time_point:
                lineage_id = lineage_id_creator.get_lineage_id(links, position)
                original_track_id = lineage_id_creator.get_original_track_id(links, position)
                cell_type_id = hash(linking_markers.get_position_type(links, position))
                density = cell_density_calculator.get_density_mm1(positions_of_time_point, position, resolution)
                times_divided = cell_division_counter.find_times_divided(links, position, first_time_point_number)
                times_neighbor_died = deaths_nearby_tracks.count_nearby_deaths_in_past(links, position)

                vector = position.to_vector_um(resolution)
                file_handle.write(f"{vector.x},{vector.y},{vector.z},{density},{times_divided},{times_neighbor_died},"
                                  f"{cell_type_id},{lineage_id},{original_track_id}\n")


def _export_colormap_file(folder: str, links: Links):
    """Writes the given Matplotlib colormap as a Paraview JSON colormap."""
    # Create colormap
    rgb_points = []
    max_track_id = links.get_highest_track_id()
    for i in range(-1, max_track_id + 1):
        # x_value goes from -1 to 1
        color = lineage_id_creator.get_color_for_lineage_id(i)
        print(i, matplotlib.colors.to_hex(color, keep_alpha=False))

        rgb_points.append(i)
        rgb_points.append(color[0])
        rgb_points.append(color[1])
        rgb_points.append(color[2])

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
