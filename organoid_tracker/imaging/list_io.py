"""Often, multiple experiments are used together for a certain analysis. You can save the path to where these
experiments are stored in an autlist file. This module contains functions to save and load Used to load multiple experiments (with images) at once."""

import json
import os
from typing import List

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.image_loading import general_image_loader
from organoid_tracker.imaging import io

FILES_LIST_EXTENSION = ".autlist"


def load_experiment_list_file(open_files_list_file: str) -> List[Experiment]:
    """Loads all the listed files in the given file.."""
    start_dir = os.path.dirname(open_files_list_file)
    with open(open_files_list_file, "r") as handle:
        experiments_json = json.load(handle)

    experiments = list()
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

        if "images_container" in experiment_json and "images_pattern" in experiment_json:
            images_container = experiment_json["images_container"]
            if not os.path.isabs(images_container):
                images_container = os.path.join(start_dir, images_container)
            images_pattern = experiment_json["images_pattern"]
            general_image_loader.load_images(experiment, images_container, images_pattern,
                                             min_time_point=min_time_point, max_time_point=max_time_point)
            loaded_anything = True

        if not loaded_anything:
            raise ValueError("No experiment defined in " + json.dumps(experiment_json) + ".")

        experiments.append(experiment)
    return experiments
