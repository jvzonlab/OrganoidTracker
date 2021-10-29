from typing import Dict, Any, Iterable, Optional

from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.gui_experiment import SingleGuiTab
from organoid_tracker.gui.window import Window
import json
import os
from typing import List

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.imaging import io
from organoid_tracker.imaging.list_io import FILES_LIST_EXTENSION, load_experiment_list_file


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Project-Tabs//Import open tabs...":
            lambda: _import_open_tabs(window),
        "File//Project-Tabs//Export open tabs...":
            lambda: _export_open_tabs(window)
    }


def to_experiment_list_file_structure(tabs: Iterable[SingleGuiTab]) -> Optional[List[Dict[str, Any]]]:
    experiments_json = list()

    for tab in tabs:
        experiment_json = dict()
        experiment = tab.experiment

        if tab.undo_redo.has_unsaved_changes() or\
                (experiment.last_save_file is None and experiment.positions.has_positions()):
            # Force a save if there are unsaved changes or if no file location is known to store the positions
            option = dialog.prompt_options(f"Save experiment \"{experiment.name}\"",
                                             f"You have unsaved changes in the experiment \"{experiment.name}\"."
                                             f"\nDo you want to save them first?",
                                  option_1="Save", option_2="Save As...", option_3="Skip experiment")
            if option == 1:
                if not action.save_tracking_data_of_tab(tab):
                    return None  # Save failed
            elif option == 2:
                if not action.save_tracking_data_of_tab(tab, force_save_as=True):
                    return None  # Save failed
            elif option == 3:
                continue  # Skip this experiment
            else:
                return None

        if experiment.last_save_file is not None:
            experiment_json["experiment_file"] = experiment.last_save_file

        images_container, images_pattern = experiment.images.image_loader().serialize_to_config()
        if len(images_container) > 0 or len(images_pattern) > 0:
            experiment_json["images_container"] = images_container
            experiment_json["images_pattern"] = images_pattern

        if experiment.first_time_point_number() is not None:
            experiment_json["min_time_point"] = experiment.first_time_point_number()
        if experiment.last_time_point_number() is not None:
            experiment_json["max_time_point"] = experiment.last_time_point_number()

        if len(experiment_json) > 0:
            # Only add if images, positions or both were stored
            experiments_json.append(experiment_json)
    return experiments_json


def _import_open_tabs(window: Window):
    file_list_files = dialog.prompt_load_multiple_files("Files to load", [("List file", "*" + FILES_LIST_EXTENSION)])
    if len(file_list_files) == 0:
        return

    experiments = list()
    for file_list_file in file_list_files:
        for experiment in load_experiment_list_file(file_list_file):
            experiments.append(experiment)
    for experiment in experiments:
        window.get_gui_experiment().add_experiment(experiment)


def _export_open_tabs(window: Window):
    files_json = to_experiment_list_file_structure(window.get_gui_experiment().get_all_tabs())
    if files_json is None:
        return  # Cancelled
    if len(files_json) == 0:
        window.set_status("No experiments to save.")
        return

    file = dialog.prompt_save_file("Saving file list", [("List file", "*" + FILES_LIST_EXTENSION)])
    if file is None:
        return
    with open(file, "w") as handle:
        json.dump(files_json, handle)
