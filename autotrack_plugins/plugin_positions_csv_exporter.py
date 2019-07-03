import os
from typing import Tuple, Dict, Any

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.gui import dialog
from autotrack.gui.window import Window





def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "File//Export-Export positions as CSV files...": lambda: _export_positions_as_csv(window),
    }

def _export_positions_as_csv(window: Window):
    experiment = window.get_experiment()
    if not experiment.positions.has_positions():
        raise UserError("No positions are found", "No annotated positions are found. Cannot export anything.")

    folder = dialog.prompt_save_file("Select a directory", [("Folder", "*")])
    if folder is None:
        return
    os.mkdir(folder)
    _export_to_csv(experiment, folder)
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


def _export_to_csv(experiment: Experiment, folder: str):
    resolution = experiment.images.resolution()
    positions = experiment.positions

    file_prefix = experiment.name.get_save_name() + ".csv."
    for time_point in positions.time_points():
        file_name = os.path.join(folder, file_prefix + str(time_point.time_point_number()))
        with open(file_name, "w") as file_handle:
            file_handle.write("x,y,z,value\n")
            for position in positions.of_time_point(time_point):
                x = position.x * resolution.pixel_size_x_um
                y = position.y * resolution.pixel_size_y_um
                z = position.z * resolution.pixel_size_z_um
                file_handle.write(f"{x},{y},{z},1\n")
