"""Generates config files for the cell tracker."""
import os
import sys
import shlex
from typing import Dict, Optional, Any

from ai_track.config import ConfigFile
from ai_track.core import UserError
from ai_track.gui import dialog
from ai_track.gui.window import Window
from ai_track.imaging import io


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Process//Standard-Train the neural network...": lambda: _generate_training_config(window),
         "Process//Standard-Detect cells in images...": lambda: _generate_detection_config(window),
    }


def _create_run_script(output_folder: str, script_name: str):
    bat_file = os.path.join(output_folder, script_name + ".bat")
    script_file = os.path.abspath(script_name + ".py")
    with open(bat_file, "w") as writer:
        writer.write(f"""@rem Automatically generated script for running {script_name}
@echo off
"{sys.executable}" "{script_file}"
pause""")


def _generate_training_config(window: Window):
    """For training the neural network."""
    experiments = list(window.get_experiments())
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

    config.get_or_default("max_training_steps", "100000")
    config.get_or_default("patch_shape", "64, 64, 32")
    config.get_or_default("output_folder", "./output")
    config.get_or_default("batch_size", "64")

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
            config.get_or_default("image_shape", f"{image_size_zyx[2]}, {image_size_zyx[1]}, {image_size_zyx[0]}")

        i = index + 1
        positions_file = f"ground_thruth_positions/positions_{i}." + io.FILE_EXTENSION
        io.save_positions_to_json(experiment, os.path.join(save_directory, positions_file))

        config.get_or_default(f"images_container_{i}", image_loader.serialize_to_config()[0])
        config.get_or_default(f"images_pattern_{i}", image_loader.serialize_to_config()[1])
        config.get_or_default(f"min_time_point_{i}", str(experiment.positions.first_time_point_number()))
        config.get_or_default(f"max_time_point_{i}", str(experiment.positions.last_time_point_number()))
        config.get_or_default(f"positions_file_{i}", positions_file)


    config.get_or_default(f"images_container_{i + 1}", "<stop>")
    config.save_if_changed()
    _create_run_script(save_directory, "ai_track_train_network")
    dialog.popup_message("Configuration files created", "The configuration files were created successfully. Please run"
                                                       " the ai_track_train_network script from that directory:"
                                                       f"\n\n{save_directory}")


def _generate_detection_config(window: Window):
    """For applying an already trained network on new images."""
    experiment = window.get_experiment()
    image_loader = experiment.images.image_loader()
    if not image_loader.has_images():
        raise UserError("No images", "No images were loaded, so no cells can be detected. Please load some images"
                                     " first.")
    resolution = experiment.images.resolution()  # Checks if resolution has been set

    checkpoint_directory =  _get_checkpoints_folder()
    if checkpoint_directory is None:
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
    config.get_or_default("pixel_size_x_um", str(resolution.pixel_size_x_um))
    config.get_or_default("pixel_size_z_um", str(resolution.pixel_size_z_um))
    config.get_or_default("time_point_duration_m", str(resolution.time_point_interval_m))
    config.get_or_default("checkpoint_folder", checkpoint_directory)
    config.save_if_changed()
    _create_run_script(save_directory, "ai_track_predict_positions")
    dialog.popup_message("Configuration file created", "The configuration file was created successfully. Please run"
                                                       " the ai_track_predict_positions script from that directory:"
                                                       f"\n\n{save_directory}")


def _get_checkpoints_folder() -> Optional[str]:
    if not dialog.popup_message_cancellable("Checkpoints folder",
                                            "First, we will ask you where you have stored the checkpoints folder, which contains the trained model."):
        return None
    while True:
        directory = dialog.prompt_directory("Please choose a checkpoint folder")
        if not directory:
            return None  # Cancelled, stop loop
        if os.path.isfile(os.path.join(directory, "checkpoint")):
            return directory  # Successful, stop loop

        # Unsuccessful
        dialog.popup_error("Not a checkpoint folder",
                           "The selected folder does not contain a trained model; it contains no 'checkpoint' file."
                           " Please select another folder. Typically, this folder is named `checkpoints`.")

