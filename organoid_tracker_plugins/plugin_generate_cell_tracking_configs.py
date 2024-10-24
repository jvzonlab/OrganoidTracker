"""Generates config files for the cell tracker."""
import json
import os
import sys
import shlex
from typing import Dict, Optional, Any, Tuple

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import io, list_io
from organoid_tracker.linking_analysis import linking_markers

_TRAINING_PATCH_SHAPE_ZYX: Tuple[int, int, int] = (32, 64, 64)
_TRAINING_PATCH_SHAPE_ZYX_DIVISION: Tuple[int, int, int] = (8, 32, 32)
_TRAINING_PATCH_SHAPE_ZYX_LINKING: Tuple[int, int, int] = (8, 32, 32)

def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Tools//Train-Train network for cell detection...": lambda: _generate_position_training_config(window),
        "Tools//Train-Train network for dividing cells...": lambda: _generate_division_training_config(window),
        "Tools//Train-Train network for linking...": lambda: _generate_link_training_config(window),
        "Tools//Use-Detect cells in images...": lambda: _generate_position_detection_config(window),
        "Tools//Use-Detect dividing cells...": lambda: _generate_division_detection_config(window),
        "Tools//Use-Detect link likelihoods...": lambda: _generate_link_detection_config(window),
        "Tools//Use-Create links between time points...": lambda: _generate_linking_config(window),
        "Tools//All-Perform all tracking steps...": lambda : _generate_all_steps_config(window),
        "Tools//Error rates-Find scaling temperature": lambda: _generate_calibrate_marginalization_config(window),
        "Tools//Error rates-Compute marginalized error rates": lambda: _generate_marginalization_config(window)
    }


def _create_run_script(output_folder: str, script_name: str):
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
set TF_FORCE_GPU_ALLOW_GROWTH=true
"{sys.executable}" "{script_file}"
pause""")

    # For Linux
    sh_file = os.path.join(output_folder, script_name + ".sh")
    with open(sh_file, "w") as writer:
        writer.write(f"""#!/bin/bash
# Automatically generated script for running {script_name}
{shlex.quote(sys.executable)} {shlex.quote(script_file)}
""")
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

    config.get_or_default("epochs", "50")
    config.get_or_default("patch_shape",
                          f"{_TRAINING_PATCH_SHAPE_ZYX[2]}, {_TRAINING_PATCH_SHAPE_ZYX[1]}, {_TRAINING_PATCH_SHAPE_ZYX[0]}")
    config.get_or_default("output_folder", "./output")
    config.get_or_default("batch_size", "30")

    config.get_or_default(f"time_window_before", str(-1))
    config.get_or_default(f"time_window_after", str(1))

    config.get_or_default(f"use_tfrecords", str(True))

    i = 0
    for index, experiment in enumerate(experiments):
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in project {experiment.name}, so it cannot be used"
                                         f" for training. Please make sure that all open projects are suitable for training.")
        if not experiment.positions.has_positions():
            raise UserError("No positions", f"No tracking data was found in project {experiment.name}, so it cannot be"
                                            f" used for training. Please make sure that all open projects are suitable for training.")

        if index == 0:
            # Save expected image shape
            image_size_zyx = image_loader.get_image_size_zyx()
            if image_size_zyx is None:
                raise UserError("Image size is not constant", f"No constant size is specified for the loaded images of"
                                                              f" project {experiment.name}. Cannot use the project for training.")

        i = index + 1
        positions_file = f"ground_truth_positions/positions_{i}.aut"
        io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))

        config.get_or_default(f"images_container_{i}", image_loader.serialize_to_config()[0])
        config.get_or_default(f"images_pattern_{i}", image_loader.serialize_to_config()[1])
        config.get_or_default(f"images_channels_{i}", "1")
        config.get_or_default(f"min_time_point_{i}", str(experiment.positions.first_time_point_number()))
        config.get_or_default(f"max_time_point_{i}", str(experiment.positions.last_time_point_number()))
        config.get_or_default(f"positions_file_{i}", positions_file)
        # new


    config.get_or_default(f"images_container_{i + 1}", "<stop>")
    config.save()
    _create_run_script(save_directory, "organoid_tracker_train_position_network")
    _popup_confirmation(save_directory, "organoid_tracker_train_position_network")


def _generate_position_detection_config(window: Window):
    """For applying an already trained network on new images."""
    experiments = list(window.get_active_experiments())
    for experiment in experiments:
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in the experiment \"{experiment.name}\","
                            f" so no cells can be detected. Please load some images first.")

    model_folder = _get_model_folder("positions")
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
    list_io.save_experiment_list_file(experiments, os.path.join(save_directory, "Dataset" + list_io.FILES_LIST_EXTENSION),
                                      tracking_files_folder=tracking_files_folder)

    config = ConfigFile("predict_positions", folder_name=save_directory)
    config.get_or_default("dataset_file", "Dataset" + list_io.FILES_LIST_EXTENSION)
    config.get_or_default("model_folder", model_folder)
    config.get_or_default("predictions_output_folder", "Nucleus center predictions")

    config.get_or_default("patch_shape_y", str(320))
    config.get_or_default("patch_shape_x", str(320))
    config.get_or_default("buffer_z", str(1))
    config.get_or_default("buffer_y", str(32))
    config.get_or_default("buffer_x", str(32))
    config.get_or_default("images_channels", str(window.display_settings.image_channel.index_one))

    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_positions")
    _popup_confirmation(save_directory, "organoid_tracker_predict_positions")


def _generate_division_training_config(window: Window):
    """For training the neural network."""
    experiments = list(window.get_active_experiments())
    if len(experiments) == 0:
        raise UserError("No experimental data loaded", "No projects are open. Please load all data (images and"
                                                       " tracking data) that you want to use for training.")

    if not dialog.popup_message_cancellable("Output folder",
                                            "All projects that are currently open will be used for training. You will"
                                            " be asked to select an output folder for training."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    config = ConfigFile("train_division_network", folder_name=save_directory)
    if len(experiments) == 1:
        if not dialog.prompt_yes_no("Experiments", "Only one project is open. Training on a single data set is not"
                                                   " recommended. For a quick test it's fine, but ideally you should have a more"
                                                   " data sets loaded.\n\nDo you want to continue with just this data set?"):
            return

    config.get_or_default("epochs", "50")
    config.get_or_default("patch_shape",
                          f"{_TRAINING_PATCH_SHAPE_ZYX_DIVISION[2]}, {_TRAINING_PATCH_SHAPE_ZYX_DIVISION[1]}, {_TRAINING_PATCH_SHAPE_ZYX_DIVISION[0]}")
    config.get_or_default("output_folder", "./output")
    config.get_or_default("batch_size", "30")

    config.get_or_default(f"time_window_before", str(-1))
    config.get_or_default(f"time_window_after", str(1))

    config.get_or_default(f"use_tfrecords", str(False))

    i = 0
    for index, experiment in enumerate(experiments):
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in project {experiment.name}, so it cannot be used"
                                         f" for training. Please make sure that all open projects are suitable for training.")
        if not experiment.positions.has_positions():
            raise UserError("No positions", f"No tracking data was found in project {experiment.name}, so it cannot be"
                                            f" used for training. Please make sure that all open projects are suitable for training.")

        if index == 0:
            # Save expected image shape
            image_size_zyx = image_loader.get_image_size_zyx()
            if image_size_zyx is None:
                raise UserError("Image size is not constant", f"No constant size is specified for the loaded images of"
                                                              f" project {experiment.name}. Cannot use the project for training.")

        i = index + 1
        positions_file = f"ground_truth_positions/positions_{i}.aut"
        io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))

        config.get_or_default(f"images_container_{i}", image_loader.serialize_to_config()[0])
        config.get_or_default(f"images_pattern_{i}", image_loader.serialize_to_config()[1])
        config.get_or_default(f"images_channels_{i}", "1")
        config.get_or_default(f"min_time_point_{i}", str(experiment.positions.first_time_point_number()))
        config.get_or_default(f"max_time_point_{i}", str(experiment.positions.last_time_point_number()))
        config.get_or_default(f"positions_file_{i}", positions_file)
        # new


    config.get_or_default(f"images_container_{i + 1}", "<stop>")
    config.save()
    _create_run_script(save_directory, "organoid_tracker_train_division_network")
    _popup_confirmation(save_directory, "organoid_tracker_train_division_network")


def _generate_division_detection_config(window: Window):
    """For applying an already trained network on new images."""
    experiment = window.get_experiment()
    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so no cells can be detected. Please load some images"
                                     " first.")

    checkpoint_directory = _get_model_folder("divisions")
    if checkpoint_directory is None:
        return

    if not dialog.popup_message_cancellable("Out folder",
                                            "Second, we will ask you to select an output folder."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    positions_file = "Input positions.aut"
    io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))

    config = ConfigFile("predict_divisions", folder_name=save_directory)
    config.get_or_default("images_container", image_loader.serialize_to_config()[0], store_in_defaults=True)
    config.get_or_default("images_pattern", image_loader.serialize_to_config()[1], store_in_defaults=True)
    config.get_or_default("positions_file", positions_file)
    config.get_or_default("min_time_point", str(image_loader.first_time_point_number()), store_in_defaults=True)
    config.get_or_default("max_time_point", str(image_loader.last_time_point_number()), store_in_defaults=True)
    config.get_or_default("checkpoint_folder", checkpoint_directory)
    config.get_or_default("predictions_output_folder", "out")
    config.get_or_default("save_video_ram", "true")
    config.get_or_default("images_channels", str(window.display_settings.image_channel.index_one))

    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_divisions")
    _popup_confirmation(save_directory, "organoid_tracker_predict_divisions")


def _generate_link_training_config(window: Window):
    """For training the neural network."""
    experiments = list(window.get_active_experiments())
    for experiment in experiments:
        experiment.images.resolution()  # Forces all experiments to have a resolution set
    if len(experiments) == 0:
        raise UserError("No experimental data loaded", "No projects are open. Please load all data (images and"
                                                       " tracking data) that you want to use for training.")

    if not dialog.popup_message_cancellable("Output folder",
                                            "All projects that are currently open will be used for training. You will"
                                            " be asked to select an output folder for training."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    config = ConfigFile("train_link_network", folder_name=save_directory)
    if len(experiments) == 1:
        if not dialog.prompt_yes_no("Experiments", "Only one project is open. Training on a single data set is not"
                                                   " recommended. For a quick test it's fine, but ideally you should have a more"
                                                   " data sets loaded.\n\nDo you want to continue with just this data set?"):
            return

    config.get_or_default("epochs", "50")
    config.get_or_default("patch_shape",
                          f"{_TRAINING_PATCH_SHAPE_ZYX_LINKING[2]}, {_TRAINING_PATCH_SHAPE_ZYX_LINKING[1]}, {_TRAINING_PATCH_SHAPE_ZYX_LINKING[0]}")
    config.get_or_default("output_folder", "./output")
    config.get_or_default("batch_size", "120")

    config.get_or_default(f"time_window_before", str(0))
    config.get_or_default(f"time_window_after", str(0))

    config.get_or_default(f"use_TFRecords", str(False))

    i = 0
    for index, experiment in enumerate(experiments):
        image_loader = experiment.images.image_loader()
        if not image_loader.has_images():
            raise UserError("No images", f"No images were loaded in project {experiment.name}, so it cannot be used"
                                         f" for training. Please make sure that all open projects are suitable for training.")
        if not experiment.positions.has_positions():
            raise UserError("No positions", f"No tracking data was found in project {experiment.name}, so it cannot be"
                                            f" used for training. Please make sure that all open projects are suitable for training.")

        if index == 0:
            # Save expected image shape
            image_size_zyx = image_loader.get_image_size_zyx()
            if image_size_zyx is None:
                raise UserError("Image size is not constant", f"No constant size is specified for the loaded images of"
                                                              f" project {experiment.name}. Cannot use the project for training.")

        i = index + 1
        positions_file = f"ground_truth_positions/positions_{i}.aut"
        io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))

        config.get_or_default(f"images_container_{i}", image_loader.serialize_to_config()[0])
        config.get_or_default(f"images_pattern_{i}", image_loader.serialize_to_config()[1])
        config.get_or_default(f"images_channels_{i}", "1")
        config.get_or_default(f"min_time_point_{i}", str(experiment.positions.first_time_point_number()))
        config.get_or_default(f"max_time_point_{i}", str(experiment.positions.last_time_point_number()))
        config.get_or_default(f"positions_file_{i}", positions_file)
        # new


    config.get_or_default(f"images_container_{i + 1}", "<stop>")
    config.save()
    _create_run_script(save_directory, "organoid_tracker_train_link_network")
    _popup_confirmation(save_directory, "organoid_tracker_train_link_network")


def _generate_link_detection_config(window: Window):
    """For applying an already trained network on new images."""
    experiment = window.get_experiment()

    # Make sure that a resolution is stored
    experiment.images.resolution()

    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so no cells can be detected. Please load some images"
                                     " first.")

    checkpoint_directory = _get_model_folder("links")
    if checkpoint_directory is None:
        return

    if not dialog.popup_message_cancellable("Out folder",
                                            "Second, we will ask you to select an output folder."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    positions_file = "Input positions.aut"
    io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))

    config = ConfigFile("predict_links", folder_name=save_directory)
    config.get_or_default("images_container", image_loader.serialize_to_config()[0], store_in_defaults=True)
    config.get_or_default("images_pattern", image_loader.serialize_to_config()[1], store_in_defaults=True)
    config.get_or_default("positions_file", positions_file)
    config.get_or_default("min_time_point", str(image_loader.first_time_point_number()), store_in_defaults=True)
    config.get_or_default("max_time_point", str(image_loader.last_time_point_number()), store_in_defaults=True)
    config.get_or_default("checkpoint_folder", checkpoint_directory)
    config.get_or_default("predictions_output_folder", "out")
    config.get_or_default("save_video_ram", "true")
    config.get_or_default("images_channels", str(window.display_settings.image_channel.index_one))

    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_links")
    _popup_confirmation(save_directory, "organoid_tracker_predict_links")


def _generate_linking_config(window: Window):
    experiment = window.get_experiment()
    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so we cannot use various heuristics to see how likely a"
                                     " cell is a dividing cell. Please load some images first.")
    if not experiment.positions.has_positions():
        raise UserError("No positions found", "No cell positions loaded. The linking algorithm links existing cell"
                                              " positions together. You can obtain cell positions using a neural"
                                              " network, see the manual.")
    experiment.images.resolution()  # Check for resolution

    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return
    positions_file = "positions." + io.FILE_EXTENSION
    io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))
    config = ConfigFile("create_links", folder_name=save_directory)
    config.get_or_default("images_container", image_loader.serialize_to_config()[0], store_in_defaults=True)
    config.get_or_default("images_pattern", image_loader.serialize_to_config()[1], store_in_defaults=True)
    config.get_or_default("min_time_point", str(image_loader.first_time_point_number()), store_in_defaults=True)
    config.get_or_default("max_time_point", str(image_loader.last_time_point_number()), store_in_defaults=True)
    config.get_or_default("positions_file", "./" + positions_file)
    config.get_or_default("output_file", "./Automatic links." + io.FILE_EXTENSION)
    config.save()
    _create_run_script(save_directory, "organoid_tracker_create_links")
    _popup_confirmation(save_directory, "organoid_tracker_create_links")


def _generate_calibrate_marginalization_config(window: Window):
    """For training the neural network."""
    experiments = list(window.get_active_experiments())
    if len(experiments) == 0:
        raise UserError("No experimental data loaded", "No projects are open. Please load all data that you want to use for calibration.")

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
    experiment = window.get_experiment()

    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so no cells can be detected. Please load some images"
                                     " first.")

    if not dialog.popup_message_cancellable("Out folder",
                                            "Select an output folder."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    checkpoint_directory = _get_model_folder("positions")
    if checkpoint_directory is None:
        return

    config = ConfigFile("predict_positions", folder_name=save_directory)

    config.get_or_default("images_container", image_loader.serialize_to_config()[0], store_in_defaults=True)
    config.get_or_default("images_pattern", image_loader.serialize_to_config()[1], store_in_defaults=True)
    config.get_or_default("min_time_point", str(image_loader.first_time_point_number()), store_in_defaults=True)
    config.get_or_default("max_time_point", str(image_loader.last_time_point_number()), store_in_defaults=True)
    config.get_or_default("images_channels", str(window.display_settings.image_channel.index_one), store_in_defaults=True)

    _pixel_size_x_um = config.get_or_default("pixel_size_x_um", "",
                                             comment="Resolution of the images. Only used if the image files and"
                                                     " tracking files don't provide a resolution.",
                                             store_in_defaults=True)
    _pixel_size_y_um = config.get_or_default("pixel_size_y_um", "", store_in_defaults=True)
    _pixel_size_z_um = config.get_or_default("pixel_size_z_um", "", store_in_defaults=True)
    _time_point_duration_m = config.get_or_default("time_point_duration_m", "", store_in_defaults=True)

    config.get_or_default("dataset_file", "")
    config.get_or_default("model_folder", checkpoint_directory)
    config.get_or_default("predictions_output_folder", "Nuclear center predictions")
    config.get_or_default("patch_shape_y", str(320))
    config.get_or_default("patch_shape_x", str(320))
    config.get_or_default("buffer_z", str(1))
    config.get_or_default("buffer_y", str(32))
    config.get_or_default("buffer_x", str(32))
    config.get_or_default("positions_output_file", "Automatic positions.aut",
                                         comment="Output file for the positions, can be viewed using the visualizer program.")

    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_positions")

    checkpoint_directory = _get_model_folder("divisions")
    if checkpoint_directory is None:
        return

    config = ConfigFile("predict_divisions", folder_name=save_directory)
    config.get_or_default("positions_file", "Automatic positions.aut")
    config.get_or_default("checkpoint_folder", checkpoint_directory)
    config.get_or_default("positions_output_file", "Automatic divisions.aut",
                                         comment="Output file for the positions, can be viewed using the visualizer program.")
    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_divisions")

    checkpoint_directory = _get_model_folder("links")
    if checkpoint_directory is None:
        return

    config = ConfigFile("predict_links", folder_name=save_directory)
    config.get_or_default("positions_file", "Automatic divisions.aut")
    config.get_or_default("checkpoint_folder", checkpoint_directory)
    config.get_or_default("positions_output_file", "Automatic links.aut",
                          comment="Output file for the positions, can be viewed using the visualizer program.")
    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_links")

    config = ConfigFile("create_links", folder_name=save_directory)
    config.get_or_default("positions_file", "Automatic links.aut")
    config.get_or_default("output_file", "_Automatic links.aut")
    config.save()
    _create_run_script(save_directory, "organoid_tracker_create_links")

    config = ConfigFile("marginalisation", folder_name=save_directory)
    config.get_or_default("solution_links_file", "clean_Automatic links.aut")
    config.get_or_default("all_links_file", "all_links_clean_Automatic links.aut")

    _create_run_script(save_directory, "organoid_tracker_marginalization")

    config.save()
    _popup_confirmation(save_directory, "executable")


def _get_model_folder(model_type: str) -> Optional[str]:
    if not dialog.popup_message_cancellable("Trained model folder",
                                            "First, we will ask you where you have stored the " + model_type + " model."):
        return None
    while True:
        directory = dialog.prompt_directory("Please choose a model folder")
        if not directory:
            return None  # Cancelled, stop loop
        if os.path.isfile(os.path.join(directory, "saved_model.pb")):

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
        else:
            # Unsuccessful
            dialog.popup_error("Not a model containing folder",
                               "The selected folder does not contain a trained model; it contains no 'saved_model.pb' file."
                               " Please select another folder. Typically, this folder is named `trained_model`.")


def _get_all_links_file() -> Optional[str]:
    if not dialog.popup_message_cancellable("All links file",
                                            "Where have you stored the file with all possible links"):
        return None
    while True:
        file = dialog.prompt_load_file('choose a file', [('aut_file', '*.aut')])
        if not file:
            return None  # Cancelled, stop loop

        if os.path.isfile(file):
            return file