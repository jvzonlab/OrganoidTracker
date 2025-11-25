"""Generates config files for the cell tracker."""
import json
import os
import shlex
import sys
from typing import Dict, Optional, Any, Tuple, List


from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError, TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.threading import Task
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import io, list_io
from organoid_tracker.visualizer import activate
from organoid_tracker.visualizer.exitable_image_visualizer import ExitableImageVisualizer

_TRAINING_PATCH_SHAPE_ZYX: Tuple[int, int, int] = (32, 96, 96)
_TRAINING_PATCH_SHAPE_ZYX_DIVISION: Tuple[int, int, int] = (12, 64, 64)
_TRAINING_PATCH_SHAPE_ZYX_LINKING: Tuple[int, int, int] = (16, 64, 64)


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Tools//Train-Train network for cell detection...": lambda: _generate_position_training_config(window),
        "Tools//Train-Train network for dividing cells...": lambda: _generate_division_training_config(window),
        "Tools//Train-Train network for linking...": lambda: _generate_link_training_config(window),
        "Tools//Use-Detect cells in images...": lambda: _generate_position_detection_config(window),
        "Tools//Use-Detect dividing cells...": lambda: _generate_division_detection_config(window),
        "Tools//Use-Detect link likelihoods...": lambda: _generate_link_detection_config(window),
        "Tools//Use-Create tracks...": lambda: _generate_tracks_config(window),
        "Tools//All-Perform all tracking steps...": lambda : _generate_all_steps_config(window),
        "Tools//Error rates-Compute marginalized error rates...": lambda: _generate_marginalization_config(window),
        "Tools//Error rates-Find scaling temperature...": lambda: _generate_calibrate_marginalization_config(window),
    }


def _create_run_script(output_folder: str, script_name: str, *, scripts_to_run: Optional[List[str]] = None):
    """Creates a Batch/Bash script to run the tracking step(s) with the given script_name. If scripts_to_run is given,
    it will run those tracking steps instead, and script_name will only be used for the name of the Batch/Bash file."""
    if scripts_to_run is None:
        scripts_to_run = [script_name]

    # For Windows
    conda_env_folder = os.sep + "envs" + os.sep
    conda_installation_folder = sys.base_exec_prefix
    if conda_env_folder in conda_installation_folder:
        conda_installation_folder = conda_installation_folder[0:conda_installation_folder.index(conda_env_folder)]
    bat_file = os.path.join(output_folder, script_name + ".bat")
    with open(bat_file, "w") as writer:
        writer.write(f"""@rem Automatically generated script for running {script_name}
@echo off
@CALL "{conda_installation_folder}\\condabin\\conda.bat" activate {os.getenv('CONDA_DEFAULT_ENV')}""")
        for i, script_to_run in enumerate(scripts_to_run):
            script_file = os.path.abspath(script_to_run + ".py")
            writer.write(f'\n"{sys.executable}" "{script_file}"')
            if i < len(scripts_to_run) - 1:
                # If a script crashes, we don't want to run the next one, so we check the error level
                writer.write("\nif errorlevel 1 goto end")
        if len(scripts_to_run) > 1:
            # Target for the above errorlevel check
            writer.write("\n:end")
        writer.write("\npause\n")

    # For Linux
    sh_file = os.path.join(output_folder, script_name + ".sh")
    with open(sh_file, "w") as writer:
        writer.write(f"""#!/bin/bash
# Automatically generated script for running {script_name}
conda activate {shlex.quote(os.getenv('CONDA_DEFAULT_ENV'))}
""")
        for script_to_run in scripts_to_run:
            script_file = os.path.abspath(script_to_run + ".py")
            writer.write(f"{shlex.quote(sys.executable)} {shlex.quote(script_file)}")
    os.chmod(sh_file, 0o777)


def _create_run_script_no_pause(output_folder: str, script_name: str):
    script_file = os.path.abspath(script_name + ".py")

    # For Windows
    conda_env_folder = os.sep + "envs" + os.sep
    conda_installation_folder = sys.base_exec_prefix
    if conda_env_folder in conda_installation_folder:
        conda_installation_folder = conda_installation_folder[0:conda_installation_folder.index(conda_env_folder)]
    bat_file = os.path.join(output_folder, script_name + ".bat")
    with open(bat_file, "w") as writer:
        writer.write(f"""@rem Automatically generated script for running {script_name}
@echo off
@CALL "{conda_installation_folder}\\condabin\\conda.bat" activate {os.getenv('CONDA_DEFAULT_ENV')}
"{sys.executable}" "{script_file}"
""")


def _popup_confirmation(output_folder: str, script_name: str, ):
    if dialog.prompt_options("Configuration files created", f"The configuration files were created successfully. Please"
                                                            f" run the {script_name} script from that directory:\n\n{output_folder}",
                             option_1="Open that directory", option_default=DefaultOption.OK) == 1:
        dialog.open_file(output_folder)


def _generate_position_training_config(window: Window):
    """For training the neural network."""
    experiments = list(window.get_active_experiments())
    _validate_experiments_for_training(experiments)
    if len(experiments) == 0:
        raise UserError("No experimental data loaded", "No projects are open. Please load all data (images and"
                                                       " tracking data) that you want to use for training.")
    if len(experiments) == 1:
        if not dialog.prompt_yes_no("Experiments", "Only one project is open. Training on a single data set is not"
                                                   " recommended. For a quick test it's fine, but ideally you should have a more"
                                                   " data sets loaded.\n\nIf you already have multiple experiments loaded, make"
                                                   " sure to select <all experiments> in the experiment selection box.\n\nDo you"
                                                   " want to continue with just this data set?"):
            return
    if not dialog.popup_message_cancellable("Output folder",
                                            f"{len(experiments)} experiment(s) will be used for training. You will"
                                            f" be asked to select an output folder for training."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    config = ConfigFile("train_position_network", folder_name=save_directory)

    config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("epochs", "50")
    config.get_or_default("patch_shape",
                          f"{_TRAINING_PATCH_SHAPE_ZYX[2]}, {_TRAINING_PATCH_SHAPE_ZYX[1]}, {_TRAINING_PATCH_SHAPE_ZYX[0]}")
    config.get_or_default("output_folder", "./output")
    config.get_or_default("batch_size", "16")

    config.get_or_default(f"time_window_before", str(0))
    config.get_or_default(f"time_window_after", str(1))

    tracking_files_folder = os.path.join(save_directory, "Ground truth files")
    os.makedirs(tracking_files_folder, exist_ok=True)
    list_io.save_experiment_list_file(experiments, os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                      tracking_files_folder=tracking_files_folder)

    config.save()
    _create_run_script(save_directory, "organoid_tracker_train_position_network")
    _popup_confirmation(save_directory, "organoid_tracker_train_position_network")


def _validate_experiments_for_training(experiments: List[Experiment]):
    """Raises UserError if the experiments are not suitable for training, because they are missing images, positions
    or a resolution."""
    for experiment in experiments:
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in project {experiment.name}, so it cannot be used"
                                         f" for training. Please make sure that all open projects are suitable for training.")
        if not experiment.positions.has_positions():
            raise UserError("No positions", f"No tracking data was found in project {experiment.name}, so it cannot be"
                                            f" used for training. Please make sure that all open projects are suitable for training.")
        image_size_zyx = image_loader.get_image_size_zyx()
        if image_size_zyx is None:
            raise UserError("Image size is not constant", f"No constant size is specified for the loaded images of"
                                                          f" project {experiment.name}. Cannot use the project for training.")
        experiment.images.resolution()  # Forces resolution check


def _generate_position_detection_config(window: Window):
    """For applying an already trained network on new images."""
    activate(_PositionPredictionVisualizer(window))


def _generate_division_training_config(window: Window):
    """For training the neural network."""
    experiments = list(window.get_active_experiments())
    _validate_experiments_for_training(experiments)
    if len(experiments) == 0:
        raise UserError("No experimental data loaded", "No projects are open. Please load all data (images and"
                                                       " tracking data) that you want to use for training.")
    if len(experiments) == 1:
        if not dialog.prompt_yes_no("Experiments", "Only one project is open. Training on a single data set is not"
                                                   " recommended. For a quick test it's fine, but ideally you should have a more"
                                                   " data sets loaded.\n\nDo you want to continue with just this data set?"):
            return
    if not dialog.popup_message_cancellable("Output folder",
                                            "All projects that are currently open will be used for training. You will"
                                            " be asked to select an output folder for training."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    # Save tracking data files
    tracking_files_folder = os.path.join(save_directory, "Ground truth files")
    os.makedirs(tracking_files_folder, exist_ok=True)
    list_io.save_experiment_list_file(experiments, os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                      tracking_files_folder=tracking_files_folder)

    # Save configuration file
    config = ConfigFile("train_division_network", folder_name=save_directory)

    config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("epochs", "50")
    config.get_or_default("patch_shape",
                          f"{_TRAINING_PATCH_SHAPE_ZYX_DIVISION[2]}, {_TRAINING_PATCH_SHAPE_ZYX_DIVISION[1]}, {_TRAINING_PATCH_SHAPE_ZYX_DIVISION[0]}")
    config.get_or_default("output_folder", "./output")
    config.get_or_default("batch_size", "16")

    config.get_or_default(f"time_window_before", str(-1))
    config.get_or_default(f"time_window_after", str(1))



    config.save()
    _create_run_script(save_directory, "organoid_tracker_train_division_network")
    _popup_confirmation(save_directory, "organoid_tracker_train_division_network")


def _generate_division_detection_config(window: Window):
    """For applying an already trained network on new images."""
    experiments = list(window.get_active_experiments())
    for experiment in experiments:
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in the experiment \"{experiment.name}\","
                                         f" so no cells can be detected. Please load some images first.")
        experiment.images.resolution()  # Check for resolution

    if not dialog.popup_message_cancellable("Trained model folder",
                                            "First, we will ask you where you have stored the divisions model."):
        return None
    model_folder = _get_model_folder("divisions")
    if model_folder is None:
        return

    if not dialog.popup_message_cancellable("Out folder",
                                            "Second, we will ask you to select an output folder."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    tracking_files_folder = os.path.join(save_directory, "Input files")
    os.makedirs(tracking_files_folder, exist_ok=True)
    list_io.save_experiment_list_file(experiments,
                                      os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                      tracking_files_folder=tracking_files_folder)

    config = ConfigFile("predict_divisions", folder_name=save_directory)
    config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("model_folder", model_folder)
    config.get_or_default("predictions_output_folder", "Division predictions")
    config.get_or_default("images_channels", str(window.display_settings.image_channel.index_one))

    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_divisions")
    _popup_confirmation(save_directory, "organoid_tracker_predict_divisions")


def _generate_link_training_config(window: Window):
    """For training the neural network."""
    experiments = list(window.get_active_experiments())
    _validate_experiments_for_training(experiments)

    if len(experiments) == 0:
        raise UserError("No experimental data loaded", "No projects are open. Please load all data (images and"
                                                       " tracking data) that you want to use for training.")
    if len(experiments) == 1:
        if not dialog.prompt_yes_no("Experiments", "Only one project is open. Training on a single data set is not"
                                                   " recommended. For a quick test it's fine, but ideally you should have a more"
                                                   " data sets loaded.\n\nDo you want to continue with just this data set?"):
            return

    if not dialog.popup_message_cancellable("Output folder",
                                            "All projects that are currently open will be used for training. You will"
                                            " be asked to select an output folder for training."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    # Save ground truth tracking data files
    tracking_files_folder = os.path.join(save_directory, "Ground truth files")
    os.makedirs(tracking_files_folder, exist_ok=True)
    list_io.save_experiment_list_file(experiments, os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                      tracking_files_folder=tracking_files_folder)

    config = ConfigFile("train_link_network", folder_name=save_directory)
    config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("epochs", "50")
    config.get_or_default("patch_shape",
                          f"{_TRAINING_PATCH_SHAPE_ZYX_LINKING[2]}, {_TRAINING_PATCH_SHAPE_ZYX_LINKING[1]}, {_TRAINING_PATCH_SHAPE_ZYX_LINKING[0]}")
    config.get_or_default("output_folder", "./output")
    config.get_or_default("batch_size", "8")

    config.get_or_default(f"time_window_before", str(0))
    config.get_or_default(f"time_window_after", str(1))

    config.save()
    _create_run_script(save_directory, "organoid_tracker_train_link_network")
    _popup_confirmation(save_directory, "organoid_tracker_train_link_network")


def _generate_link_detection_config(window: Window):
    """For applying an already trained network on new images."""
    experiments = list(window.get_active_experiments())
    for experiment in experiments:
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in the experiment \"{experiment.name}\","
                                         f" so no cells can be detected. Please load some images first.")
        experiment.images.resolution()  # Check for resolution

    if not dialog.popup_message_cancellable("Trained model folder",
                                            "First, we will ask you where you have stored the links model."):
        return None
    model_folder = _get_model_folder("links")
    if model_folder is None:
        return

    if not dialog.popup_message_cancellable("Out folder",
                                            "Second, we will ask you to select an output folder."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    tracking_files_folder = os.path.join(save_directory, "Input files")
    os.makedirs(tracking_files_folder, exist_ok=True)
    list_io.save_experiment_list_file(experiments,
                                      os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                      tracking_files_folder=tracking_files_folder)

    config = ConfigFile("predict_links", folder_name=save_directory)
    config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("model_folder", model_folder)
    config.get_or_default("predictions_output_folder", "Link predictions")
    config.get_or_default("images_channels", str(window.display_settings.image_channel.index_one))

    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_links")
    _popup_confirmation(save_directory, "organoid_tracker_predict_links")


def _generate_tracks_config(window: Window):
    experiments = list(window.get_active_experiments())
    for experiment in experiments:
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in the experiment \"{experiment.name}\","
                                         f" so no cells can be detected. Please load some images first.")
        if not experiment.links.has_links():
            raise UserError("No links", f"No tracking data was found in project {experiment.name}. Did you"
                                        f"run all the previous tracking steps?")
        experiment.images.resolution()  # Check for resolution

    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    tracking_files_folder = os.path.join(save_directory, "Input files")
    os.makedirs(tracking_files_folder, exist_ok=True)
    list_io.save_experiment_list_file(experiments,
                                      os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                      tracking_files_folder=tracking_files_folder)

    config = ConfigFile("create_tracks", folder_name=save_directory)
    config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("output_folder", "Output tracks")

    config.save()
    _create_run_script(save_directory, "organoid_tracker_create_tracks")
    _popup_confirmation(save_directory, "organoid_tracker_create_tracks")


def _generate_calibrate_marginalization_config(window: Window):
    """For training the neural network."""
    experiments = list(window.get_active_experiments())
    if len(experiments) == 0:
        raise UserError("No experimental data loaded",
                        "No projects are open. Please load all data that you want to use for calibration.")

    if not dialog.popup_message_cancellable("Output folder",
                                            "All projects that are currently open will be used for training. You will"
                                            " be asked to select an output folder for calibration."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    config = ConfigFile("find_scaling_temperature", folder_name=save_directory)
    if len(experiments) == 1:
        if not dialog.prompt_yes_no("Experiments", "Only one project is open. Calibration on a single data set is not"
                                                   " recommended. For a quick test it's fine, but ideally you should have a more"
                                                   " data sets loaded.\n\nDo you want to continue with just this data set?"):
            return

    config.get_or_default("output_folder", "./output")
    _steps = int(config.get_or_default("size subset (steps away from link of interest)", str(3)))

    i = 0
    for index, experiment in enumerate(experiments):

        if not experiment.positions.has_positions():
            raise UserError("No positions", f"No tracking data was found in project {experiment.name}, so it cannot be"
                                            f" used for calibration. Please make sure that all open projects are suitable for training.")

        i = index + 1
        positions_file = f"ground_truth_positions/all_links_{i}.aut"
        io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))

        config.get_or_default(f"min_time_point_{i}", str(experiment.positions.first_time_point_number()))
        config.get_or_default(f"max_time_point_{i}", str(experiment.positions.last_time_point_number()))
        config.get_or_default(f"all_links_{i}", positions_file)
        # new

    config.get_or_default(f"images_container_{i + 1}", "<stop>")
    config.save()
    _create_run_script(save_directory, "organoid_tracker_calibrate_marginalization")
    _popup_confirmation(save_directory, "organoid_tracker_calibrate_marginalization")


def _generate_marginalization_config(window: Window):
    """For applying an already trained network on new images."""
    experiment = window.get_experiment()

    all_links_file = _get_all_links_file()
    if all_links_file is None:
        return

    if not dialog.popup_message_cancellable("Out folder",
                                            "Second, we will ask you to select an output folder."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    positions_file = "solution_link_file.aut"
    io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))

    config = ConfigFile("marginalisation", folder_name=save_directory)
    config.get_or_default("solution_links_file", positions_file)
    config.get_or_default("all_links_file", all_links_file)
    config.get_or_default("min_time_point", str(experiment.first_time_point_number()), store_in_defaults=True)
    config.get_or_default("max_time_point", str(experiment.last_time_point_number()), store_in_defaults=True)
    config.get_or_default("predictions_output_folder", "out")

    config.save()
    _create_run_script(save_directory, "organoid_tracker_marginalization")
    _popup_confirmation(save_directory, "organoid_tracker_marginalization")


def _generate_all_steps_config(window: Window):
    """For applying an already trained network on new images."""
    experiments = list(window.get_active_experiments())
    for experiment in experiments:
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in the experiment \"{experiment.name}\","
                            f" so no cells can be detected. Please load some images first.")
        experiment.images.resolution()  # Check for resolution

    if not dialog.popup_message_cancellable("Trained models folder",
                                            "First, we will ask you where you have stored the trained models"
                                            " for positions, divisions and links, in that order. "):
        return None
    model_folder_positions = _get_model_folder("positions")
    if model_folder_positions is None:
        return
    model_folder_divisions = _get_model_folder("divisions")
    if model_folder_divisions is None:
        return
    model_folder_links = _get_model_folder("links")
    if model_folder_links is None:
        return

    if not dialog.popup_message_cancellable("Out folder",
                                            "Great, that were all models! Now we will ask you to select an output folder."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    tracking_files_folder = os.path.join(save_directory, "Input files")
    os.makedirs(tracking_files_folder, exist_ok=True)
    list_io.save_experiment_list_file(experiments, os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                      tracking_files_folder=tracking_files_folder)

    # Generate one big config file for all steps
    config = ConfigFile("predict_positions", folder_name=save_directory)
    config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("model_folder", model_folder_positions)
    config.get_or_default("predictions_output_folder", "Nucleus center predictions")
    config.get_or_default("positions_output_folder", "Automatic positions")
    config.get_or_default("patch_shape_y", str(320))
    config.get_or_default("patch_shape_x", str(320))
    config.get_or_default("buffer_z", str(1))
    config.get_or_default("buffer_y", str(32))
    config.get_or_default("buffer_x", str(32))
    config.get_or_default("images_channels", str(window.display_settings.image_channel.index_one), store_in_defaults=True)
    config.save()
    config = ConfigFile("predict_divisions", folder_name=save_directory)
    config.get_or_default("dataset_file", "Automatic positions/_All" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("model_folder", model_folder_divisions)
    config.get_or_default("predictions_output_folder", "Division predictions")
    config.save()
    config = ConfigFile("predict_links", folder_name=save_directory)
    config.get_or_default("dataset_file", "Division predictions/_All" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("model_folder", model_folder_links)
    config.get_or_default("predictions_output_folder", "Link predictions")
    config.save()
    config = ConfigFile("create_tracks", folder_name=save_directory)
    config.get_or_default("dataset_file", "Link predictions/_All" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("output_folder", "Output tracks")
    config.save()

    _create_run_script(save_directory, "organoid_tracker_run_all_tracking_steps", scripts_to_run=[
        "organoid_tracker_predict_positions", "organoid_tracker_predict_divisions", "organoid_tracker_predict_links",
        "organoid_tracker_create_tracks"
    ])
    _popup_confirmation(save_directory, "organoid_tracker_run_all_tracking_steps")


def _get_model_folder(model_type: str) -> Optional[str]:
    while True:
        directory = dialog.prompt_directory(f"Please choose a {model_type} model folder")
        if not directory:
            return None  # Cancelled, stop loop
        if os.path.isfile(os.path.join(directory, "model.keras")):
            if os.path.isfile(os.path.join(directory, "settings.json")):
                with open(os.path.join(directory, "settings.json")) as file_handle:
                    found_model_type = json.load(file_handle)["type"]
                    if found_model_type == model_type:
                        return directory  # Successful, stop loop
                    else:
                        dialog.popup_error("This is a " + found_model_type + " model",
                                           "The selected folder contains a model for dealing with " + found_model_type + ". For this analysis step, we need a " + model_type + " model.")
            else:
                dialog.popup_error("Not an OrganoidTracker model",
                                   "The selected folder does not contain a `settings.json` file."
                                   "Therefore, this model is not compatible with OrganoidTracker.")
        elif os.path.isfile(os.path.join(directory, "saved_model.pb")):
            dialog.popup_error("Old model", "This model was written for Tensorflow 2.7 and is no longer"
                                            " compatible with OrganoidTracker.")
        else:
            # Unsuccessful
            dialog.popup_error("Not a model-containing folder",
                               "The selected folder does not contain a trained model; it contains no 'model.keras' file."
                               " Please select another folder. Typically, this folder is named `model_" + model_type + "`.")


def _get_all_links_file() -> Optional[str]:
    if not dialog.popup_message_cancellable("All links file",
                                            "Where have you stored the file with all possible links?"):
        return None
    while True:
        file = dialog.prompt_load_file('choose a file', [('aut_file', '*.aut')])
        if not file:
            return None  # Cancelled, stop loop

        if os.path.isfile(file):
            return file


class _PositionPredictionVisualizer(ExitableImageVisualizer):
    """Set a model folder in the Parameters menu. If you're happy with how the predictions look, generate config files
    and start the full predictions using the Edit menu."""

    model_folder: Optional[str] = None
    min_quantile: float = 0.01
    max_quantile: float = 0.99
    xy_scaling: float = 1.0
    z_scaling: float = 1.0
    channels: List[ImageChannel]
    _predicted_positions: List[Position] = None

    def __init__(self, window: Window):
        super().__init__(window)
        self.channels = [window.display_settings.image_channel]
        self._predicted_positions = list()

    def _draw_positions(self):
        max_intensity_projection = self._display_settings.max_intensity_projection

        positions_x_list, positions_y_list, positions_edge_colors, positions_edge_widths, positions_marker_sizes = \
            list(), list(), list(), list(), list()

        min_z, max_z = self._z - self.MAX_Z_DISTANCE, self._z + self.MAX_Z_DISTANCE
        for position in self._predicted_positions:
            if not max_intensity_projection and (position.z < min_z or position.z > max_z):
                continue
            if position.time_point_number() != self._time_point.time_point_number():
                continue
            dz = self._z - round(position.z)

            # Add marker
            edge_color, edge_width = self._get_position_edge(position)

            positions_x_list.append(position.x)
            positions_y_list.append(position.y)
            positions_edge_colors.append(edge_color)
            positions_edge_widths.append(edge_width)
            dz_penalty = 0 if dz == 0 or max_intensity_projection else abs(dz) + 1
            positions_marker_sizes.append(max(1, 7 - dz_penalty + edge_width) ** 2)

        self._ax.scatter(positions_x_list, positions_y_list, s=positions_marker_sizes, facecolor="#be2edd",
                         edgecolors="black", linewidths=1, marker="s")

    def get_extra_menu_options(self) -> Dict[str, Any]:
        return {
            **super().get_extra_menu_options(),
            "Parameters//Quantile-Set min intensity quantile...": self._set_min_quantile,
            "Parameters//Quantile-Set max intensity quantile...": self._set_max_quantile,
            "Parameters//Scaling-Set XY scaling...": self._set_xy_scaling,
            "Parameters//Scaling-Set Z scaling...": self._set_z_scaling,
            "Parameters//Model-Set image channel for prediction...": self._set_channel,
            "Parameters//Model-Set model folder...": self._set_model_folder,
            "Edit//Model-Generate configuration files...": self._generate_config,
            "Edit//Model-Test on this time point...": self._test_on_time_point,
        }

    def _get_figure_title(self) -> str:
        return "Position detection\nTime point " + str(self._time_point.time_point_number()) + "    (z=" + \
                      self._get_figure_title_z_str() + ")"

    def _test_on_time_point(self):
        self._experiment.images.resolution()  # Make sure a resolution is set
        if not self._experiment.images.image_loader().has_images():
            # Nothing to predict
            self._predicted_positions.clear()
            return
        if self.model_folder is None:
            # No model selected
            self._predicted_positions.clear()
            raise UserError("No model set", "Set a model folder in the Parameters menu to enable position predictions.")

        if self._window.get_scheduler().has_active_tasks():
            return
        self.update_status("Predicting positions...")
        self._window.get_scheduler().add_task(_PredictPositions(self, self._experiment, self._time_point))

    def _generate_config(self):
        experiments = list(self._window.get_active_experiments())
        for experiment in experiments:
            image_loader = experiment.images.image_loader()
            if not image_loader.has_images():
                raise UserError("No images", f"No images were loaded in the experiment \"{experiment.name}\","
                                             f" so no cells can be detected. Please load some images first.")
            experiment.images.resolution()  # Checks for resolution

        if self.model_folder is None:
            raise UserError("No model folder selected", "Please select a model folder in the Parameters"
                                                        " menu before generating the configuration files.")

        save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
        if save_directory is None:
            return

        tracking_files_folder = os.path.join(save_directory, "Input files")
        os.makedirs(tracking_files_folder, exist_ok=True)
        list_io.save_experiment_list_file(experiments,
                                          os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                          tracking_files_folder=tracking_files_folder)

        config = ConfigFile("predict_positions", folder_name=save_directory)
        config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
        config.get_or_default("model_folder", self.model_folder)
        config.get_or_default("predictions_output_folder", "Nucleus center predictions")
        config.get_or_default("patch_shape_y", str(320))
        config.get_or_default("patch_shape_x", str(320))
        config.get_or_default("buffer_z", str(1))
        config.get_or_default("buffer_y", str(32))
        config.get_or_default("buffer_x", str(32))
        config.get_or_default("scale_factor_xy", str(self.xy_scaling))
        config.get_or_default("scale_factor_z", str(self.z_scaling))
        config.get_or_default("intensity_min_quantile", str(self.min_quantile))
        config.get_or_default("intensity_max_quantile", str(self.max_quantile))
        config.get_or_default("images_channels", ",".join(str(channel.index_one) for channel in self.channels))

        config.save()
        _create_run_script(save_directory, "organoid_tracker_predict_positions")
        _popup_confirmation(save_directory, "organoid_tracker_predict_positions")

    def _set_min_quantile(self):
        value = dialog.prompt_float("Set minimum intensity quantile",
                                    "Please enter the minimum intensity quantile to use for normalization (between 0 and 1):",
                                    default=self.min_quantile, minimum=0.0, maximum=1.0, decimals=3)
        if value is not None:
            self.min_quantile = value
            self.update_status("Set minimum intensity quantile to " + str(self.min_quantile) + ". Use the Edit menu to generate predictions.")

    def _set_max_quantile(self):
        value = dialog.prompt_float("Set maximum intensity quantile",
                                    "Please enter the maximum intensity quantile to use for normalization (between 0 and 1):",
                                    default=self.max_quantile, minimum=0.0, maximum=1.0, decimals=3)
        if value is not None:
            self.max_quantile = value
            self.update_status("Set maximum intensity quantile to " + str(self.max_quantile) + ". Use the Edit menu to generate predictions.")

    def _set_xy_scaling(self):
        value = dialog.prompt_float("Set XY scaling factor",
                                    "Please enter the scaling factor to use for the X and Y dimensions."
                                    " For example, if set to 0.5, all images are halved in size in X and Y before prediction.",
                                    default=self.xy_scaling, minimum=0.01, maximum=10.0, decimals=2)
        if value is not None:
            self.xy_scaling = value
            self.update_status("Set XY scaling factor to " + str(self.xy_scaling) + ". Use the Edit menu to generate predictions.")

    def _set_z_scaling(self):
        value = dialog.prompt_float("Set Z scaling factor",
                                    "Please enter the scaling factor to use for the Z dimension."
                                    " For example, if set to 2.0, all images are doubled in size in Z before prediction.",
                                    default=self.z_scaling, minimum=0.01, maximum=10.0, decimals=2)
        if value is not None:
            self.z_scaling = value
            self.update_status("Set Z scaling factor to " + str(self.z_scaling) + ". Use the Edit menu to generate predictions.")

    def _set_channel(self):
        available_channels_count = self._find_available_image_channels_count()
        options = [f"Channel {i}" for i in range(1, available_channels_count + 1)]
        channels = option_choose_dialog.prompt_list_multiple("Image channel for prediction",
                                             "Please select the image channel(s) that should be used for cell detection.",
                                                            label="Channels",
                                             options=options)
        if channels is not None:
            self.channels = [ImageChannel(index_zero=channel) for channel in channels]
            self.update_status("Set image channel for prediction to " + (",".join(str(channel.index_one) for channel in self.channels)) + ". Use the Edit menu to generate predictions.")

    def _set_model_folder(self):
        self.model_folder = _get_model_folder("positions")
        if self.model_folder is not None:
            self.update_status("Set model folder to " + str(self.model_folder) + ". Use the Edit menu to generate predictions.")

    def _find_available_image_channels_count(self):
        max_channel_count = 1
        for experiment in self._window.get_active_experiments():
            max_channel_count = max(max_channel_count, len(experiment.images.get_channels()))
        return max_channel_count

    def callback_position_predictions(self, time_point: TimePoint, result: List[Position]):
        if time_point != self._time_point:
            return  # Outdated result
        self._predicted_positions = result
        self.draw_view()
        self.update_status("Updated position predictions. If you're happy with them, generate configuration files using the Edit menu.")


class _PredictPositions(Task):
    _visualizer: _PositionPredictionVisualizer

    _experiment: Optional[Experiment]  # Cleared after self.compute()
    _time_point: TimePoint
    _progress: float

    def __init__(self, visualizer: _PositionPredictionVisualizer,
                 experiment: Experiment, time_point: TimePoint):
        super().__init__()
        self._visualizer = visualizer
        self._experiment = Experiment()
        self._experiment.images = experiment.images
        self._time_point = time_point
        self._progress = 0

    def _set_progress(self, progress: float):
        self._progress = progress

    def compute(self) -> Any:
        import _keras_environment
        _keras_environment.activate()
        import keras
        from organoid_tracker.neural_network.position_detection_cnn import position_predictor
        model = position_predictor.load_position_model(self._visualizer.model_folder)
        model.predict_positions(self._experiment, time_points=[self._time_point],
                                scale_factors_zyx=(self._visualizer.z_scaling, self._visualizer.xy_scaling, self._visualizer.xy_scaling),
                                intensity_quantiles=(self._visualizer.min_quantile, self._visualizer.max_quantile),
                                image_channels=set(self._visualizer.channels),
                                progress_callback=self._set_progress)
        # Free some memory, these models can be quite large
        del model
        keras.backend.clear_session()
        return list(self._experiment.positions.of_time_point(self._time_point))

    def on_finished(self, result: Any):
        self._visualizer.callback_position_predictions(self._time_point, result)

    def get_percentage_completed(self) -> Optional[int]:
        if self._progress == 0:
            return None
        return int(self._progress * 100)