"""Generates config files for the cell tracker."""
import os
from typing import Dict, Optional, Any

from autotrack.config import ConfigFile
from autotrack.core import UserError
from autotrack.gui import dialog
from autotrack.gui.window import Window


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
         "Process//Standard-Detect cells in images...": lambda: _generate_detection_config(window),
    }

def _generate_detection_config(window: Window):
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
    config.get_or_default("pixel_size_x_um", str(resolution.pixel_size_x_um))
    config.get_or_default("pixel_size_z_um", str(resolution.pixel_size_z_um))
    config.get_or_default("time_point_duration_m", str(resolution.time_point_interval_m))
    config.get_or_default("checkpoints_folder", checkpoint_directory)
    config.save_if_changed()
    dialog.popup_message("Configuration file created", "The configuration file was created successfully. Please run"
                                                       " the autotrack_predict_positions script from that directory:"
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

