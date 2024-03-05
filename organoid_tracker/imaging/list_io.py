"""Often, multiple experiments are used together for a certain analysis. You can save the path to where these
experiments are stored in an autlist file. This module contains functions to save and load Used to load multiple experiments (with images) at once."""

import json
import os
from typing import List, Iterable, Dict, Any

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.imaging import io

FILES_LIST_EXTENSION = ".autlist"


def load_experiment_list_file(open_files_list_file: str) -> Iterable[Experiment]:
    """Loads all the listed files in the given file.."""
    open_files_list_file = os.path.abspath(open_files_list_file)

    # Makes paths to images and AUT files relative to the list file, which is probably what you want
    previous_working_directory = os.getcwd()
    start_dir = os.path.dirname(open_files_list_file)
    if start_dir:
        os.chdir(start_dir)

    try:
        with open(open_files_list_file, "r", encoding="utf-8") as handle:
            experiments_json = json.load(handle)

        for experiment_json in experiments_json:
            experiment = Experiment()
            min_time_point = 0
            max_time_point = 100000000

            if "min_time_point" in experiment_json:
                min_time_point = int(experiment_json["min_time_point"])
            if "max_time_point" in experiment_json:
                max_time_point = int(experiment_json["max_time_point"])

            loaded_anything = False
            if "experiment_file" in experiment_json:
                experiment_file = experiment_json["experiment_file"]
                if not os.path.isabs(experiment_file):
                    experiment_file = os.path.join(start_dir, experiment_file)
                if not os.path.exists(experiment_file):
                    raise ValueError("File \"" + experiment_file + "\" does not exist.")
                io.load_data_file(experiment_file, experiment=experiment,
                                  min_time_point=min_time_point, max_time_point=max_time_point)
                loaded_anything = True

            if _contains_images(experiment_json):
                general_image_loader.load_images_from_dictionary(experiment, experiment_json,
                                                 min_time_point=min_time_point, max_time_point=max_time_point)
                loaded_anything = True

            if not loaded_anything:
                raise ValueError("No experiment defined in " + json.dumps(experiment_json) + ".")

            yield experiment
    finally:
        # Move back to old working directory
        os.chdir(previous_working_directory)


def _contains_images(experiment_json: Dict[str, Any]) -> bool:
    """Returns True if any of the keys starts with "images_". """
    for key in experiment_json.keys():
        if key.startswith("images_"):
            return True
    return False


def count_experiments_in_list_file(open_files_list_file: str) -> int:
    """Counts the number of experiments in the given file. Used as a quick way to get that number, instead of loading
    everything."""
    with open(open_files_list_file, "r", encoding="utf-8") as handle:
        experiments_json = json.load(handle)
        if not isinstance(experiments_json, list):
            raise TypeError("Expected a list in " + open_files_list_file + ", got " + repr(experiments_json))
        return len(experiments_json)


def save_experiment_list_file(experiments: List[Experiment], json_file_name: str):
    """Saves references to the given experiments to a file. The file will contain the paths to the last saved file of
    the experiment. Note: this method does not check for unsaved changes in the experiments, so make sure that all files
    are saved.

    Raises ValueError if the experiment contains positions or splines, but no value for experiment.last_save_file.

    For an interactive version where the user is prompted to save unsaved changes, see
    `action.to_experiment_list_file_structure(...)`.
    """
    experiments_json = []
    for experiment in experiments:
        experiment_json = dict()

        # Store experiment file
        if experiment.last_save_file is not None:
            experiment_json["experiment_file"] = experiment.last_save_file
        else:
            if experiment.positions.has_positions() or experiment.splines.has_splines():
                raise ValueError(f"The experiment \"{experiment.name}\" has not been saved to disk.")

        # Store images
        experiment_json.update(experiment.images.image_loader().serialize_to_dictionary())

        # Store time points
        if experiment.first_time_point_number() is not None:
            experiment_json["min_time_point"] = experiment.first_time_point_number()
        if experiment.last_time_point_number() is not None:
            experiment_json["max_time_point"] = experiment.last_time_point_number()

        if len(experiment_json) > 0:
            # Only add if images, positions or both were stored
            experiments_json.append(experiment_json)

    with open(json_file_name, "w", encoding="utf-8") as handle:
        json.dump(experiments_json, handle, indent=4, sort_keys=True)
