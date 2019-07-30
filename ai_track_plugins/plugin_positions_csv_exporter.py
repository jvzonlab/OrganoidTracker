import os
from typing import Tuple, Dict, Any

from ai_track.core import UserError
from ai_track.core.experiment import Experiment

from ai_track.core.links import Links
from ai_track.core.position_collection import PositionCollection
from ai_track.core.resolution import ImageResolution
from ai_track.gui import dialog
from ai_track.gui.threading import Task
from ai_track.gui.window import Window
from ai_track.position_analysis import cell_curvature_calculator


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
        window.get_scheduler().add_task(_AsyncExporter(experiment, folder))
    else:
        _write_positions_to_csv(experiment, folder)
        _export_help_file(folder)
        dialog.popup_message("Positions", "Exported all positions, as well as a help file with instructions on how to"
                             " visualize the points in Paraview.")


def _export_help_file(folder: str):
    text = """
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


class _AsyncExporter(Task):
    _positions: PositionCollection
    _links: Links
    _resolution: ImageResolution
    _folder: str
    _save_name: str

    def __init__(self, experiment: Experiment, folder: str):
        self._positions = experiment.positions.copy()
        self._links = experiment.links.copy()
        self._resolution = experiment.images.resolution()
        self._folder = folder
        self._save_name = experiment.name.get_save_name()

    def compute(self) -> Any:
        _write_positions_and_metadata_to_csv(self._positions, self._links, self._resolution, self._folder, self._save_name)
        _export_help_file(self._folder)
        return "done"  # We're not using the result

    def on_finished(self, result: Any):
        dialog.popup_message("Positions", "Exported all positions, as well as a help file with instructions on how to"
                                          " visualize the points in Paraview.")


def _write_positions_and_metadata_to_csv(positions: PositionCollection, links: Links, resolution: ImageResolution, folder: str, save_name: str):
    from ai_track.linking_analysis import lineage_id_creator
    from ai_track.position_analysis import cell_density_calculator

    file_prefix = save_name + ".csv."
    for time_point in positions.time_points():
        file_name = os.path.join(folder, file_prefix + str(time_point.time_point_number()))
        with open(file_name, "w") as file_handle:
            file_handle.write("x,y,z,density_mm1,curvature_angle,lineage_id\n")
            positions_of_time_point = positions.of_time_point(time_point)
            for position in positions_of_time_point:
                lineage_id = lineage_id_creator.get_lineage_id(links, position)
                curvature = cell_curvature_calculator.get_curvature_angle(positions, position, resolution)
                density = cell_density_calculator.get_density_mm1(positions_of_time_point, position, resolution)

                vector = position.to_vector_um(resolution)
                file_handle.write(f"{vector.x},{vector.y},{vector.z},{density},{curvature},{lineage_id}\n")


