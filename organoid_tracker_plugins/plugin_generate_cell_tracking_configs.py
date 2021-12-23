"""Generates config files for the cell tracker."""
import os
import sys
import shlex
from typing import Dict, Optional, Any, Tuple

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging import io
from organoid_tracker.linking_analysis import linking_markers

_TRAINING_PATCH_SHAPE_ZYX: Tuple[int, int, int] = (32, 64, 64)
_TRAINING_PATCH_SHAPE_ZYX_DIVISION: Tuple[int, int, int] = (8, 32, 32)
_TRAINING_PATCH_SHAPE_ZYX_LINKING: Tuple[int, int, int] = (8, 32, 32)

def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Tools//Train-Train network for cell detection...": lambda: _generate_training_config(window),
        "Tools//Train-Train network for dividing cells...": lambda: _generate_division_training_config(window),
        "Tools//Train-Train network for linking...": lambda: _generate_link_training_config(window),
        "Tools//Use-Detect cells in images...": lambda: _generate_detection_config(window),
        "Tools//Use-Detect dividing cells...": lambda: _generate_division_detection_config(window),
        "Tools//Use-Detect link likelihoods...": lambda: _generate_link_detection_config(window),
        "Tools//Use-Create links between time points...": lambda: _generate_linking_config(window),
        "Tools//Process-Detect shapes using Gaussian fit...": lambda: _generate_gaussian_fit_configs(window),
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


def _popup_confirmation(output_folder: str, script_name: str, ):
    if dialog.prompt_options("Configuration files created", f"The configuration files were created successfully. Please"
                             f" run the {script_name} script from that directory:\n\n{output_folder}",
                             option_1="Open that directory", option_default=DefaultOption.OK) == 1:
        dialog.open_file(output_folder)


def _generate_training_config(window: Window):
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

    config = ConfigFile("train_network", folder_name=save_directory)

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
    _create_run_script(save_directory, "organoid_tracker_train_network")
    _popup_confirmation(save_directory, "organoid_tracker_train_network")


def _generate_detection_config(window: Window):
    """For applying an already trained network on new images."""
    experiment = window.get_experiment()
    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so no cells can be detected. Please load some images"
                                     " first.")

    model_folder = _get_model_folder()
    if model_folder is None:
        return

    if not dialog.popup_message_cancellable("Out folder",
                                            "Second, we will ask you to select an output folder."):
        return
    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return

    config = ConfigFile("predict_positions", folder_name=save_directory)
    config.get_or_default("images_container", image_loader.serialize_to_config()[0], store_in_defaults=True)
    config.get_or_default("images_pattern", image_loader.serialize_to_config()[1], store_in_defaults=True)
    config.get_or_default("min_time_point", str(image_loader.first_time_point_number()), store_in_defaults=True)
    config.get_or_default("max_time_point", str(image_loader.last_time_point_number()), store_in_defaults=True)
    config.get_or_default("model_folder", model_folder)
    config.get_or_default("predictions_output_folder", "out")

    config.get_or_default("patch_shape_z", str(30))
    config.get_or_default("patch_shape_y", str(240))
    config.get_or_default("patch_shape_x", str(240))
    config.get_or_default("buffer_z", str(1))
    config.get_or_default("buffer_y", str(8))
    config.get_or_default("buffer_x", str(8))

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

    config = ConfigFile("train_network", folder_name=save_directory)
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
    _create_run_script(save_directory, "organoid_tracker_train_division_network")
    _popup_confirmation(save_directory, "organoid_tracker_train_division_network")


def _generate_division_detection_config(window: Window):
    """For applying an already trained network on new images."""
    experiment = window.get_experiment()
    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so no cells can be detected. Please load some images"
                                     " first.")

    #checkpoint_directory = _get_checkpoints_folder()
    checkpoint_directory = _get_model_folder()
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

    config.get_or_default("patch_shape_z", str(8))
    config.get_or_default("patch_shape_y", str(32))
    config.get_or_default("patch_shape_x", str(32))

    config.get_or_default(f"time_window_before", str(-1))
    config.get_or_default(f"time_window_after", str(1))

    config.get_or_default("save_video_ram", "true")

    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_divisions")
    _popup_confirmation(save_directory, "organoid_tracker_predict_divisions")


def _generate_link_training_config(window: Window):
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

    config = ConfigFile("train_network", folder_name=save_directory)
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
    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so no cells can be detected. Please load some images"
                                     " first.")

    #checkpoint_directory = _get_checkpoints_folder()
    checkpoint_directory = _get_model_folder()
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

    config.get_or_default("patch_shape_z", str(8))
    config.get_or_default("patch_shape_y", str(48))
    config.get_or_default("patch_shape_x", str(48))

    config.get_or_default(f"time_window_before", str(0))
    config.get_or_default(f"time_window_after", str(0))

    config.get_or_default("save_video_ram", "true")
    config.get_or_default("save_video_ram", "true")

    config.save()
    _create_run_script(save_directory, "organoid_tracker_predict_links")
    _popup_confirmation(save_directory, "organoid_tracker_predict_links")


def _generate_gaussian_fit_configs(window: Window):
    experiment = window.get_experiment()
    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so we have nothing to fit the Gaussian functions to."
                                     " Please load some images first.")
    resolution = experiment.images.resolution()
    if not experiment.positions.has_positions():
        raise UserError("No positions", "No positions were found. The Gaussian fit requires starting points as seeds."
                                        " You can obtain points manually or using the convolutional neural network. See"
                                        " the automatic tracking tutorial in the help files for more information.")

    # Erosion
    erode_option = dialog.prompt_options("Cell overlap", "How much visual overlap do you have in your images between"
                                         " the visible objects (nuclei, cells, whatever you are imaging)?\n"
                                         "\nOption 1: almost none at all"
                                         "\nOption 2: a bit"
                                         "\nOption 3: a lot of objects are connected with neighbor objects"
                                         "\n\nThe quality of the Gaussian fit is the best if you pick option 1."
                                         " However, if all your objects overlap, then it will try to fit Gaussian"
                                         " functions for all of them together, which is computationally very"
                                         " expensive.", option_1="Option 1", option_2="Option 2", option_3="Option 3")
    if erode_option is None:
        return
    print(erode_option)
    erode_passes = 0
    if erode_option == 2:
        erode_passes = 2
    elif erode_option == 3:
        erode_passes = 3

    save_directory = dialog.prompt_save_file("Output directory", [("Folder", "*")])
    if save_directory is None:
        return
    positions_file = "positions." + io.FILE_EXTENSION
    io.save_data_to_json(experiment, os.path.join(save_directory, positions_file))

    config = ConfigFile("detect_gaussian_shapes", folder_name=save_directory)
    config.get_or_default("images_container", image_loader.serialize_to_config()[0], store_in_defaults=True)
    config.get_or_default("images_pattern", image_loader.serialize_to_config()[1], store_in_defaults=True)
    config.get_or_default("min_time_point", str(image_loader.first_time_point_number()), store_in_defaults=True)
    config.get_or_default("max_time_point", str(image_loader.last_time_point_number()), store_in_defaults=True)
    config.get_or_default("positions_input_file", "./" + positions_file)
    config.get_or_default("positions_and_fit_output_file", "./Gaussian fitted positions." + io.FILE_EXTENSION)
    config.get_or_default("adaptive_threshold_block_size", str(51))
    config.get_or_default("cluster_detection_erosion_rounds", str(erode_passes))
    config.get_or_default("gaussian_fit_smooth_size", str(7))
    config.save()
    _create_run_script(save_directory, "organoid_tracker_detect_gaussian_shapes")
    _popup_confirmation(save_directory, "organoid_tracker_detect_gaussian_shapes")


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


def _get_model_folder() -> Optional[str]:
    if not dialog.popup_message_cancellable("Trained model folder",
                                            "First, we will ask you where you have stored the model."):
        return None
    while True:
        directory = dialog.prompt_directory("Please choose a model folder")
        if not directory:
            return None  # Cancelled, stop loop
        if os.path.isfile(os.path.join(directory, "saved_model.pb")):
            return directory  # Successful, stop loop

        # Unsuccessful
        dialog.popup_error("Not a model containing folder",
                           "The selected folder does not contain a trained model; it contains no 'saved_model.pb' file."
                           " Please select another folder. Typically, this folder is named `trained_model`.")

