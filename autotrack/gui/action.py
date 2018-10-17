import re
import sys
import tkinter
from os import path
from tkinter import messagebox
from typing import Optional

from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.core.links import LinkType
from autotrack.gui import Window, dialog
from autotrack.gui.dialog import popup_message_cancellable
from autotrack.imaging import tifffolder, io
from autotrack.manual_tracking import links_extractor
from autotrack.visualizer import activate
from autotrack.visualizer.empty_visualizer import EmptyVisualizer


def ask_exit(root: tkinter.Tk):
    """Exits the main window."""
    if messagebox.askyesno("Confirmation",
                           "Are you sure you want to quit the program? Any unsaved changes will be lost."):
        root.quit()
        root.destroy()


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
    if messagebox.askyesno("Confirmation",
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
        messagebox.showerror("Could not read file pattern", "Could not find 't01' (or similar) in the file name \"" +
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
        ("Detection or linking files", "*.json")])
    if file_name is None:
        return  # Cancelled

    try:
        new_experiment = io.load_data_file(file_name)
    except ValueError as e:
        messagebox.showerror("Error loading data file",
                             "Failed to load data file.\n\n" + _error_message(e))
    else:
        # Transfer image loader from old experiment
        new_experiment.image_loader(window.get_experiment().image_loader())

        window.set_experiment(new_experiment)
        window.refresh()


def add_positions(window: Window):
    experiment = window.get_experiment()

    cell_file = dialog.prompt_load_file("Select positions file", [("JSON file", "*.json")])
    if not cell_file:
        return  # Cancelled

    try:
        io.load_positions_and_shapes_from_json(experiment, cell_file)
    except ValueError as e:
        raise UserError("Error loading positions",
                        "Failed to load positions.\n\n" + _error_message(e))
    else:
        window.refresh()


def add_links(window: Window):
    experiment = window.get_experiment()

    link_file = dialog.prompt_load_file("Select link file", [("JSON file", "*.json")])
    if not link_file:
        return  # Cancelled

    try:
        io.load_linking_result(experiment, str(link_file), LinkType.BASELINE)
    except ValueError as e:
        raise UserError("Error loading links", "Failed to load links. Are you sure that is a valid JSON links"
                                               " file? Are the corresponding cell positions loaded?\n\n"
                                               + _error_message(e))
    else:
        window.refresh()


def add_guizela_tracks(window: Window):
    """Loads the tracks in Guizela's format."""
    folder = dialog.prompt_directory("Open folder with Guizela's tracks...")
    if not folder:
        return  # Cancelled

    graph = links_extractor.extract_from_tracks(folder)

    experiment = window.get_experiment()
    for particle in graph.nodes():
        experiment.add_particle(particle)
    experiment.links.add_links(LinkType.BASELINE, graph)
    window.refresh()


def export_positions_and_shapes(experiment: Experiment):
    positions_file = dialog.prompt_save_file("Save positions as...", [("JSON file", "*.json")])
    if not positions_file:
        return  # Cancelled
    io.save_positions_and_shapes_to_json(experiment, positions_file)


def export_links(experiment: Experiment):
    links = experiment.links.baseline
    if not links:
        messagebox.showerror("No links", "Cannot export links; there are no links created.")
        return

    links_file = dialog.prompt_save_file("Save links as...", [("JSON file", "*.json")])
    if not links_file:
        return  # Cancelled

    io.save_links_to_json(links, links_file)


def save_tracking_data(experiment: Experiment):
    data_file = dialog.prompt_save_file("Save data as...", [
        (io.FILE_EXTENSION.upper() + " file", "*." + io.FILE_EXTENSION)])
    if not data_file:
        return  # Cancelled

    io.save_data_to_json(experiment, data_file)


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
