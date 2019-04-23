import re
import sys
from enum import Enum
from os import path
from typing import Optional

from PyQt5.QtWidgets import QApplication
from matplotlib.figure import Figure

from autotrack.core import UserError
from autotrack.core.experiment import Experiment
from autotrack.gui import dialog, image_resolution_dialog, option_choose_dialog
from autotrack.gui.dialog import popup_message_cancellable
from autotrack.gui.gui_experiment import GuiExperiment
from autotrack.gui.undo_redo import UndoRedo
from autotrack.gui.window import Window
from autotrack.imaging import tifffolder, io, lif, liffile
from autotrack.linking_analysis import linking_markers
from autotrack.visualizer import activate
from autotrack.visualizer.empty_visualizer import EmptyVisualizer


def ask_save_unsaved_changes(gui_experiment: GuiExperiment) -> bool:
    """If there are any unsaved changes, this method will prompt the user to save them. Returns True if the user either
    successfully saved the data, or if the user doesn't want to save. Returns False if the action must be aborted."""
    if not gui_experiment.undo_redo.has_unsaved_changes():
        return True

    answer = dialog.prompt_yes_no_cancel("Confirmation", "There are unsaved changes to the tracking data. Do you want"
                                                         " to save those first?")
    if answer.is_yes():
        if save_tracking_data(gui_experiment):
            return True
    elif answer.is_no():
        return True
    return False


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
    if not ask_save_unsaved_changes(window.get_gui_experiment()):
        return  # Cancelled

    visualizer = EmptyVisualizer(window)
    activate(visualizer)
    window.get_gui_experiment().add_experiment(Experiment())


def close_experiment(window: Window):
    """Closes the current experiment."""
    if not ask_save_unsaved_changes(window.get_gui_experiment()):
        return  # Cancelled
    window.get_gui_experiment().remove_experiment(window.get_experiment())


def load_images(window: Window):
    # Show an OK/cancel box, but with an INFO icon instead of a question mark
    if not popup_message_cancellable("Image loading",
                                     "Images are expected to be 3D grayscale files. Both TIF series and LIF files are "
                                     "supported.\n\n"
                                     "For TIF files, the program expects a series of files, each file representing a "
                                     "single time point. Select the file of the first time point, which must have "
                                     "\"t1\", \"t01\", \"_1.\" or similar in the file name."):
        return  # Cancelled
    full_path = dialog.prompt_load_file("Select first image file", [
        ("TIF file", "*.tif"),
        ("TIFF file", "*.tiff"),
        ("LIF file", "*.lif")])
    if not full_path:
        return  # Cancelled
    directory, file_name = path.split(full_path)

    if file_name.endswith(".lif"):
        # LIF file loading
        from autotrack.imaging import liffile
        reader = lif.Reader(full_path)
        series = [header.getDisplayName() for header in reader.getSeriesHeaders()]
        serie_index = option_choose_dialog.popup_image_getter("Choose an image serie", "Choose an image serie", "Image serie:", series)
        if serie_index is not None:
            liffile.load_from_lif_reader(window.get_experiment().images, full_path, reader, serie_index)
            window.redraw_all()
        return

    file_name_pattern = _find_pattern(file_name)
    if file_name_pattern is None:
        dialog.popup_error("Could not read file pattern", "Could not find 't01' (or similar) in the file name \"" +
                           file_name + "\". Make sure you selected the first image.")
        return

    # Load and show images
    tifffolder.load_images_from_folder(window.get_experiment(), directory, file_name_pattern)
    window.redraw_all()


def _find_pattern(file_name: str) -> Optional[str]:
    # Support t001
    counting_part = re.search('t0*1', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return file_name[0:start] + "t%0" + str(end - start - 1) + "d" + file_name[end:]

    # Support T001
    counting_part = re.search('T0*1', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return file_name[0:start] + "T%0" + str(end - start - 1) + "d" + file_name[end:]

    # Support _001.
    counting_part = re.search('_0*1\.', file_name)
    if counting_part is not None:
        start, end = counting_part.start(0), counting_part.end(0)
        return file_name[0:start] + "_%0" + str(end - start - 2) + "d." + file_name[end:]

    return None


def load_tracking_data(window: Window):
    if not ask_save_unsaved_changes(window.get_gui_experiment()):
        return  # Cancelled

    file_name = dialog.prompt_load_file("Select data file", [
        (io.FILE_EXTENSION.upper() + " file", "*." + io.FILE_EXTENSION),
        ("Detection or linking files", "*.json"),
        ("Guizela's tracking files", "lineages.p")])
    if file_name is None:
        return  # Cancelled


    if window.get_experiment().positions.has_positions():
        # Need to load data in a separate tab, copy over images
        new_experiment = io.load_data_file(file_name)
        new_experiment.images.image_loader(window.get_experiment().images.image_loader().copy())
        window.get_gui_experiment().add_experiment(new_experiment)
    else:
        # Fine to dump tracking data in existing experiment
        io.load_data_file(file_name, experiment=window.get_experiment())
        window.redraw_all()


def export_positions_and_shapes(experiment: Experiment):
    positions_file = dialog.prompt_save_file("Save positions as...", [("JSON file", "*.json")])
    if not positions_file:
        return  # Cancelled
    io.save_positions_and_shapes_to_json(experiment, positions_file)


def export_links_guizela(experiment: Experiment):
    links = experiment.links
    if not links.has_links():
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
    guizela_data_exporter.export_links(links, experiment.images.offsets, links_folder, comparisons_folder)


def save_tracking_data(gui_experiment: GuiExperiment) -> bool:
    data_file = dialog.prompt_save_file("Save data as...", [
        (io.FILE_EXTENSION.upper() + " file", "*." + io.FILE_EXTENSION)])
    if not data_file:
        return False # Cancelled

    io.save_data_to_json(gui_experiment.experiment, data_file)
    gui_experiment.undo_redo.mark_everything_saved()
    return True


def reload_plugins(window: Window):
    """Reloads all active plugins from disk."""
    count = window.reload_plugins()
    window.redraw_all()
    window.set_status(f"Reloaded all {count} plugins.")


def _error_message(error: Exception):
    return str(type(error).__name__) + ": " + str(error)


def show_manual():
    dialog.popup_manual()


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


def set_image_resolution(window: Window):
    image_resolution_dialog.popup_resolution_setter(window.get_gui_experiment())


def view_statistics(window: Window):
    experiment = window.get_experiment()
    if experiment.last_time_point_number() is None:
        raise UserError("Statistics", "No data is loaded. Cannot calculate statistics.")
    time_point_count = experiment.last_time_point_number() - experiment.first_time_point_number() + 1
    position_count = len(experiment.positions)
    links_count = len(experiment.links)
    errors_count = sum(1 for error in linking_markers.find_errored_positions(experiment.links))
    errors_percentage = errors_count/position_count*100 if position_count > 0 else 0
    dialog.popup_message("Statistics", f"There are {time_point_count} time points loaded. {position_count} positions "
                                       f" are annotated and {links_count} links have been created."
                                       f"\n\nThere are {errors_count} warnings and errors remaining for you to look at,"
                                       f" so {errors_percentage:.02f}% of all positions has an error.")


def ask_exit(gui_experiment: GuiExperiment):
    """Asks to save unsaved changes, then exits."""
    if ask_save_unsaved_changes(gui_experiment):
        QApplication.quit()
