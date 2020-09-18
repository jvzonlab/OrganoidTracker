from os import path
from typing import Optional

from PySide2.QtWidgets import QApplication
from matplotlib.figure import Figure

from organoid_tracker.core import UserError
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.gui import dialog, image_resolution_dialog, option_choose_dialog
from organoid_tracker.gui.dialog import popup_message_cancellable
from organoid_tracker.gui.gui_experiment import GuiExperiment
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import io
from organoid_tracker.imaging.image_file_name_pattern_finder import find_time_and_channel_pattern
from organoid_tracker.linking_analysis import linking_markers
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.empty_visualizer import EmptyVisualizer


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
    full_path = dialog.prompt_load_file("Select first image file", [
        ("Single TIF or TIF series", "*.tif;*.tiff"),
        ("Image per time point", "*.png;*.jpg;*.gif"),
        ("LIF file", "*.lif"),
        ("ND2 file", "*.nd2")])
    if not full_path:
        return  # Cancelled
    directory, file_name = path.split(full_path)

    if file_name.endswith(".lif"):
        # LIF file loading
        from organoid_tracker.image_loading import _lif, liffile_image_loader
        reader = _lif.Reader(full_path)
        series = [header.getName() for header in reader.getSeriesHeaders()]
        series_index = option_choose_dialog.popup_image_getter("Choose an image serie", "Choose an image serie", "Image serie:", series)
        if series_index is not None:
            liffile_image_loader.load_from_lif_reader(window.get_experiment().images, full_path, reader, series_index)
            window.redraw_all()
        return

    if file_name.endswith(".nd2"):
        # ND2 file loading
        from organoid_tracker.image_loading import nd2file_image_loader
        reader = nd2file_image_loader.Nd2File(full_path)
        max_location = reader.get_location_counts()
        location = dialog.prompt_int("Image series", f"Which image series do you want load? (1-"
                                     f"{max_location}, inclusive)", minimum=1, maximum=max_location)
        if location is not None:
            loader = nd2file_image_loader.load_image_series(reader, location)
            window.get_experiment().images.image_loader(loader)
            window.redraw_all()
        return

    file_name_pattern = find_time_and_channel_pattern(file_name)
    if file_name_pattern is None:
        file_name_pattern = file_name  # Don't use a pattern if not available
        file_name_lower = file_name.lower()
        if file_name_lower.endswith(".tif") or file_name_lower.endswith(".tiff"):
            # Try as TIF container
            from organoid_tracker.image_loading import merged_tiff_image_loader
            merged_tiff_image_loader.load_from_tif_file(window.get_experiment().images, full_path)
            window.redraw_all()
            return
        dialog.popup_message("Could not read file pattern", "Could not find 't01' (or similar) in the file name \"" +
                             file_name + "\", so only one image is loaded. If you want to load a time lapse, see the"
                             " manual for supported image formats.")

    # Load and show images
    from organoid_tracker.image_loading import folder_image_loader
    folder_image_loader.load_images_from_folder(window.get_experiment(), directory, file_name_pattern)
    window.redraw_all()


def load_tracking_data(window: Window):
    if not ask_save_unsaved_changes(window.get_gui_experiment()):
        return  # Cancelled

    file_name = dialog.prompt_load_file("Select data file", io.SUPPORTED_IMPORT_FILES)
    if file_name is None:
        return  # Cancelled

    # Replace the existing experiment with one with the same images, but the new data
    new_experiment = io.load_data_file(file_name)
    new_experiment.images.use_image_loader_from(window.get_experiment().images)
    window.get_gui_experiment().replace_selected_experiment(new_experiment)


def export_positions(experiment: Experiment):
    if not popup_message_cancellable("Notice", "This export option is intended for use with the neural network. As"
                                     " such, it uses the raw pixel coordinates (if you have specified image offsets,"
                                     " those will be ignored) and dead cells are omitted.\n\nNote that the "
                                     + io.FILE_EXTENSION + " files generated by this program are JSON files, so they're"
                                     " perfectly fine as an export format themselves. If you need something simpler,"
                                     " use the CSV export."):
        return  # Cancelled
    positions_file = dialog.prompt_save_file("Save positions as...", [("JSON file", "*.json")])
    if not positions_file:
        return  # Cancelled
    io.save_positions_to_json(experiment, positions_file)


def export_links_guizela(experiment: Experiment):
    if not experiment.links.has_links():
        raise UserError("No links", "Cannot export to this file format; there are no links created.")

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

    from organoid_tracker.guizela_tracker_compatibility import guizela_data_exporter
    guizela_data_exporter.export_links(experiment, links_folder, comparisons_folder)


def export_links_ctc(experiment: Experiment):
    if experiment.images.image_loader().get_image_size_zyx() is None:
        raise UserError("No images found", "Couldn't find an image size. Note that this data format saves the tracking"
                                           " positions in images of the same size as the original microscopy images."
                                           " Please load those first.")
    if not experiment.links.has_links():
        raise UserError("No links found", "This save format can only save tracks. Currently, there are no links"
                                          " loaded, so we have no tracks, and therefore we cannot save anything.")
    tracks_folder = dialog.prompt_save_file("Save tracks as...",
                                            [("_GT or _RES folder", "*")])
    if tracks_folder is None:
        return

    from organoid_tracker.imaging import ctc_io
    ctc_io.save_data_files(experiment, tracks_folder)


def save_tracking_data(gui_experiment: GuiExperiment) -> bool:
    data_file = dialog.prompt_save_file("Save data as...", [
        (io.FILE_EXTENSION.upper() + " file", "*." + io.FILE_EXTENSION)])
    if not data_file:
        return False # Cancelled

    io.save_data_to_json(gui_experiment.get_experiment(), data_file)
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
                                  "Developed by Rutger Kok in February 2018 - present. Copyright AMOLF.\n"
                                  "Splines and alternative data format by Guizela Huelsz-Prince\n"
                                  "Lineage tree drawing by Jeroen van Zon\n"
                                  "Convolutional neural network by Laetitia Hebert and Greg Stephens\n\n"
                                  "Various open source packages are used - see their licences, which"
                                  " you have agreed to when you used Anaconda to install them.")


class _RenameAction(UndoableAction):

    _old_name: Optional[str]
    _new_name: str

    def __init__(self, old_name: Optional[str], new_name: str):
        self._old_name = old_name
        self._new_name = new_name

    def do(self, experiment: Experiment) -> str:
        experiment.name.set_name(self._new_name)
        return f"Changed name of experiment to \"{self._new_name}\""

    def undo(self, experiment: Experiment) -> str:
        experiment.name.set_name(self._old_name)
        if self._old_name is None:
            return f"Removed name \"{self._new_name}\" from the expermient"
        else:
            return f"Changed name of the experiment back to \"{self._old_name}\""


def rename_experiment(window: Window):
    experiment = window.get_experiment()
    name = dialog.prompt_str("Name of the experiment", "Enter a new name for the experiment.",
                             default=str(experiment.name))
    if name:
        window.perform_data_action(_RenameAction(experiment.name.get_name(), name))


def set_image_resolution(window: Window):
    image_resolution_dialog.popup_resolution_setter(window)


def view_statistics(window: Window):
    experiment = window.get_experiment()
    if experiment.last_time_point_number() is None:
        raise UserError("Statistics", "No data is loaded. Cannot calculate statistics.")
    time_point_count = experiment.last_time_point_number() - experiment.first_time_point_number() + 1
    position_count = len(experiment.positions)
    links_count = len(experiment.links)
    errors_count = sum(1 for error in linking_markers.find_errored_positions(experiment.position_data))
    errors_percentage = errors_count/position_count*100 if position_count > 0 else 0
    dialog.popup_message("Statistics", f"There are {time_point_count} time points loaded. {position_count} positions "
                                       f" are annotated and {links_count} links have been created."
                                       f"\n\nThere are {errors_count} warnings and errors remaining for you to look at,"
                                       f" so {errors_percentage:.02f}% of all positions has an error.")


def ask_exit(gui_experiment: GuiExperiment):
    """Asks to save unsaved changes, then exits."""
    if ask_save_unsaved_changes(gui_experiment):
        QApplication.quit()
