import re
import sys
from os import path
from typing import Optional

from PyQt5.QtWidgets import QApplication
from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.gui import Window, dialog
from autotrack.gui.dialog import popup_message_cancellable
from autotrack.imaging import tifffolder, io
from autotrack.visualizer import activate
from autotrack.visualizer.empty_visualizer import EmptyVisualizer


def ask_exit(experiment: Experiment):
    """Exits the main window, after asking if the user wants to save."""
    answer = dialog.prompt_yes_no_cancel("Confirmation", "Do you want to save your changes first?")
    if answer.is_yes():
        if save_tracking_data(experiment):
            QApplication.quit()
    elif answer.is_no():
        QApplication.quit()


def toggle_axis(figure: Figure):
    """Toggles whether the axes are visible."""
    set_visible = None
    for axis in figure.axes:
        if set_visible is None:
            set_visible = not axis.get_xaxis().get_visible()
        axis.get_xaxis().set_visible(set_visible)
        axis.get_yaxis().set_visible(set_visible)
    figure.canvas.draw()


def new(window: Window):
    """Starts a new experiment."""
    if dialog.prompt_yes_no("Confirmation",
                            "Are you sure you want to start a new project? Any unsaved changed will be lost."):
        window.set_experiment(Experiment())
        visualizer = EmptyVisualizer(window)
        activate(visualizer)


def load_images(window: Window):
    # Show an OK/cancel box, but with an INFO icon instead of a question mark
    if not popup_message_cancellable("Image loading", "Images are expected to be 3D grayscale TIF files. Each TIF file "
                               "represents a single time point.\n\n"
                               "Please select the TIF file of the first time point. The file name of the image must "
                               "contain \"t1\", \"t01\", \"_1.\" or similar in the file name."):
        return  # Cancelled
    full_path = dialog.prompt_load_file("Select first image file", [
        ("TIF file", "*.tif"),
        ("TIFF file", "*.tiff")])
    if not full_path:
        return  # Cancelled
    directory, file_name = path.split(full_path)
    file_name_pattern = _find_pattern(file_name)
    if file_name_pattern is None:
        dialog.popup_error("Could not read file pattern", "Could not find 't01' (or similar) in the file name \"" +
                           file_name + "\". Make sure you selected the first image.")
        return

    # Load and show images
    tifffolder.load_images_from_folder(window.get_experiment(), directory, file_name_pattern)
    window.refresh()


def _find_pattern(file_name: str) -> Optional[str]:
    # Support t001
    counting_part = re.search('t0*1', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return file_name[0:start] + "t%0" + str(end - start - 1) + "d" + file_name[end:]

    # Support _001.
    counting_part = re.search('_0*1\.', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return file_name[0:start] + "_%0" + str(end - start - 2) + "d." + file_name[end:]

    return None


def load_tracking_data(window: Window):
    file_name = dialog.prompt_load_file("Select data file", [
        (io.FILE_EXTENSION.upper() + " file", "*." + io.FILE_EXTENSION),
        ("Detection or linking files", "*.json"),
        ("Guizela's tracking files", "*.p")])
    if file_name is None:
        return  # Cancelled

    new_experiment = io.load_data_file(file_name)
    # Transfer image loader from old experiment
    new_experiment.image_loader(window.get_experiment().image_loader())

    window.set_experiment(new_experiment)
    window.refresh()


def export_positions_and_shapes(experiment: Experiment):
    positions_file = dialog.prompt_save_file("Save positions as...", [("JSON file", "*.json")])
    if not positions_file:
        return  # Cancelled
    io.save_positions_and_shapes_to_json(experiment, positions_file)


def export_links(experiment: Experiment):
    links = experiment.links.graph
    if not links:
        raise UserError("No links", "Cannot export links; there are no links created.")

    links_file = dialog.prompt_save_file("Save links as...", [("JSON file", "*.json")])
    if not links_file:
        return  # Cancelled

    io.save_links_to_json(links, links_file)


def export_links_guizela(experiment: Experiment):
    links = experiment.links.graph
    if not links:
        raise UserError("No links", "Cannot export links; there are no links created.")

    links_folder = dialog.prompt_save_file("Save links as...", [("Folder", "*")])
    if not links_folder:
        return  # Cancelled
    comparisons_folder = None
    if dialog.prompt_yes_no("Track ids",
                                   "Do you want to reuse existing track ids? This is useful for comparing data.\n\n If"
                                   " yes, then you will be asked to select the folder containing the existing tracks."):
        comparisons_folder = dialog.prompt_directory("Select folder with track ids to reuse")
        if not comparisons_folder:
            return

    from autotrack.manual_tracking import guizela_data_exporter
    guizela_data_exporter.export_links(links, links_folder, comparisons_folder)


def save_tracking_data(experiment: Experiment) -> bool:
    data_file = dialog.prompt_save_file("Save data as...", [
        (io.FILE_EXTENSION.upper() + " file", "*." + io.FILE_EXTENSION)])
    if not data_file:
        return False # Cancelled

    io.save_data_to_json(experiment, data_file)
    return True


def reload_plugins(window: Window):
    """Reloads all active plugins from disk."""
    count = window.reload_plugins()
    window.refresh()
    window.set_status(f"Reloaded all {count} plugins.")


def _error_message(error: Exception):
    return str(type(error).__name__) + ": " + str(error)


def show_manual():
    dialog.open_file(path.join(path.dirname(path.abspath(sys.argv[0])), "manuals", "VISUALIZER.pdf"))


def about_the_program():
    dialog.popup_message("About", "Cell detection and linking.\n\n"
                                  "Originally developed by Rutger Kok in February - July 2018. Copyright AMOLF.\n\n"
                                  "Various open source packages are used - see their licences, which"
                                  " you have agreed to when you used Anaconda to install them.")


def rename_experiment(window: Window):
    experiment = window.get_experiment()
    name = dialog.prompt_str("Name of the experiment", "Enter a new name for the experiment.",
                             default=str(experiment.name))
    if name:
        experiment.name.set_name(name)
        window.set_status("Changed the name of the experiment to \"" + str(experiment.name) + "\".")
