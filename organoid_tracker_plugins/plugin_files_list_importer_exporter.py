import json
from typing import Dict, Any

from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging.list_io import FILES_LIST_EXTENSION, load_experiment_list_file


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "File//Project-Tabs//Import open tabs...":
            lambda: _import_open_tabs(window),
        "File//Project-Tabs//Export open tabs...":
            lambda: _export_open_tabs(window)
    }


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
    files_json = action.to_experiment_list_file_structure(window.get_gui_experiment().get_all_tabs())
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
