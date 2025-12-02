"""Often, multiple experiments are used together for a certain analysis. You can save the path to where these
experiments are stored in an autlist file. This module contains functions to save and load Used to load multiple experiments (with images) at once."""

import json
import os
from typing import List, Iterable, Dict, Any, Optional

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.imaging import io

FILES_LIST_EXTENSION = ".autlist"


def load_experiment_list_file(open_files_list_file: str, *, load_images: bool = True,
                              min_time_point: int = -100000000, max_time_point: int = 100000000) -> Iterable[Experiment]:
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
            min_time_point_experiment = min_time_point
            max_time_point_experiment = max_time_point

            # If min and max time points are specified in the experiment file, restrict the range
            if "min_time_point" in experiment_json:
                min_time_point_experiment = max(min_time_point, int(experiment_json["min_time_point"]))
            if "max_time_point" in experiment_json:
                max_time_point_experiment = min(max_time_point, int(experiment_json["max_time_point"]))

            loaded_anything = False
            if "experiment_file" in experiment_json:
                experiment_file = experiment_json["experiment_file"]
                if not os.path.isabs(experiment_file):
                    experiment_file = os.path.join(start_dir, experiment_file)
                if not os.path.exists(experiment_file):
                    raise ValueError("File \"" + experiment_file + "\" does not exist.")
                io.load_data_file(experiment_file, experiment=experiment,
                                  min_time_point=min_time_point_experiment, max_time_point=max_time_point_experiment)
                loaded_anything = True

            if load_images and _contains_images(experiment_json):
                general_image_loader.load_images_from_dictionary(experiment, experiment_json,
                                                                 min_time_point=min_time_point_experiment, max_time_point=max_time_point_experiment)
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


def save_experiment_list_file(experiments: List[Experiment], json_file_name: str, *,
                              tracking_files_folder: Optional[str] = None, append_to_file: bool = False) -> None:
    """Saves references to the given experiments to a file.

    If a folder has been specified for tracking_files_folder, the experiments will be saved to that folder.

    Otherwise, the file will reference the current location where the tracking files are stored, taken from
    experiment.last_save_file. This method does not check for unsaved changes in the experiments. Raises ValueError if
    the experiment contains positions or splines, but no value for experiment.last_save_file. For an interactive version
    where the user is prompted to save unsaved changes, see `action.to_experiment_list_file_structure(...)`.

    When saving many experiments at once, it can become expensive to keep them all in memory. In that case, you can
    save them one by one to the list file, by setting append_to_file=True. For the first iteration though, make sure
    that the file does not exist yet or that you set append_to_file=False, otherwise the old file will keep on growing.
    """
    save_base_folder = os.path.dirname(os.path.abspath(json_file_name))

    # Load existing file if append_to_file is True, otherwise start with an empty list
    if append_to_file and os.path.exists(json_file_name):
        with open(json_file_name, "r", encoding="utf-8") as handle:
            experiments_json = json.load(handle)
            if not isinstance(experiments_json, list):
                raise TypeError("Expected a list in " + json_file_name + ", got " + repr(experiments_json))
    else:
        experiments_json = list()

    for experiment in experiments:
        i = len(experiments_json)  # Current index in the list file
        experiment_json = dict()

        # Store experiment file
        if tracking_files_folder is not None:
            # Save file to designated folder
            file_name = os.path.join(tracking_files_folder, f"{i + 1}. {experiment.name.get_save_name()}." + io.FILE_EXTENSION)
            io.save_data_to_json(experiment, file_name)
            experiment_json["experiment_file"] = _relpath(file_name, start=save_base_folder)
        else:
            # Just enter the location where the file is currently saved
            if experiment.last_save_file is not None:
                # Make last_save_file relative to save_base_folder
                experiment_json["experiment_file"] = _relpath(experiment.last_save_file, start=save_base_folder)
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


def _relpath(path: str, start: str) -> str:
    """Returns the relative path from start to path, but if that is not possible (on different drives on Windows),
    returns the absolute path."""
    try:
        return os.path.relpath(path, start=start)
    except ValueError:
        return os.path.abspath(path)
