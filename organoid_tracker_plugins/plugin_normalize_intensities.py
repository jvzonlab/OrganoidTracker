"""If the organoid has a membrane marker, then that can be used for segmentation."""
from typing import Dict, Any, Set, List

from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog, option_choose_dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.position_analysis import intensity_calculator


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Intensity//Record-Normalize intensities//Normalize with background and z correction...":
            lambda: _normalize_with_background_and_z(window),
        "Intensity//Record-Normalize intensities//Normalize with background correction...":
            lambda: _normalize_with_background(window),
        "Intensity//Record-Normalize intensities//Normalize with time correction...":
            lambda: _normalize_with_time(window),
        "Intensity//Record-Normalize intensities//Normalize without corrections...":
            lambda: _normalize_without_background(window),
        "Intensity//Record-Normalize intensities//Remove normalization...":
            lambda: _remove_normalization(window)
    }


def _get_all_intensity_keys(window: Window) -> Set[str]:
    """Gets all intensity keys available for all experiments"""
    keys = set()
    for experiment in window.get_active_experiments():
        keys |= set(intensity_calculator.get_intensity_keys(experiment))
    return keys


def _verify_saved_intensities(window: Window):
    """Raises a UserError if no intensities were stored."""
    if len(_get_all_intensity_keys(window)) == 0:
        raise UserError("No intensities", "No intensities were measured. Please do so first.")


def _prompt_intensity_keys(window: Window) -> List[str]:
    """If there are more than one intensity keys, this prompts the user which ones should be used. Returns an empty list
     if the user pressed Cancel, or if there were no intensities selected."""
    intensity_keys = list(_get_all_intensity_keys(window))

    if len(intensity_keys) > 1:
        intensity_key_indices = option_choose_dialog.prompt_list_multiple("Intensities", "We found multiple intensities. Which"
                                                                   " ones should we normalize? Select all that apply",
                                                                   "Intensity keys:", intensity_keys)
        if intensity_key_indices is None:
            return []  # Cancelled
        if len(intensity_key_indices) == 0:
            # User pressed OK, but didn't select anything. Likely in error, so notify the user.
            raise UserError("No keys selected", "No intensity keys were selected. Please check the boxes of the"
                                                " intensities that you want to normalize.")
        return [intensity_keys[i] for i in intensity_key_indices]
    return intensity_keys

def _normalize_with_background_and_z(window: Window):
    _verify_saved_intensities(window)
    if not dialog.popup_message_cancellable("Normalization", "The normalization of the intensities will be changed. "
                                            "The lowest found intensity in the experiment is used for setting the "
                                            "background. In addition, the intensities will be multiplied to obtain "
                                            "a median intensity of 1 at each z position."):
        return

    intensity_keys = _prompt_intensity_keys(window)
    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        for intensity_key in intensity_keys:
            intensity_calculator.perform_intensity_normalization(experiment, background_correction=True,
                                                                 z_correction=True, intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()


def _normalize_with_background(window: Window):
    _verify_saved_intensities(window)
    if not dialog.popup_message_cancellable("Normalization", "The normalization of the intensities will be changed.\n"
                                            "The lowest found intensity in the experiment is used for setting the\n"
                                            "background. In addition, the intensities will be multiplied to obtain\n"
                                            "an overall median intensity of 1."):
        return

    intensity_keys = _prompt_intensity_keys(window)
    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        for intensity_key in intensity_keys:
            intensity_calculator.perform_intensity_normalization(experiment, background_correction=True,
                                                                 z_correction=False, intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()


def _normalize_with_time(window: Window):
    _verify_saved_intensities(window)
    if not dialog.popup_message_cancellable("Normalization", "The normalization of the intensities will be changed.\n"
                                            "The intensities will be multiplied to obtain\n"
                                            "a median intensity of 1 at every time point."):
        return

    intensity_keys = _prompt_intensity_keys(window)
    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        for intensity_key in intensity_keys:
            intensity_calculator.perform_intensity_normalization(experiment, background_correction=False,
                                                                 z_correction=False, time_correction=True,
                                                                 intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()



def _normalize_without_background(window: Window):
    _verify_saved_intensities(window)
    if not dialog.popup_message_cancellable("Normalization", "All intensities will be normalized. No background\n"
                                            "correction is used. Still, the intensities will be multiplied to\n"
                                            "obtain an overall median intensity of 1."):
        return

    intensity_keys = _prompt_intensity_keys(window)
    for tab in window.get_gui_experiment().get_active_tabs():
        experiment = tab.experiment
        for intensity_key in intensity_keys:
            intensity_calculator.perform_intensity_normalization(experiment, background_correction=False,
                                                                 z_correction=False, intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()

def _remove_normalization(window: Window):
    _verify_saved_intensities(window)
    if not dialog.popup_message_cancellable("Normalization", "The normalization will be removed, so that script will\n"
                                                             "use the raw values again."):
        return

    intensity_keys = _prompt_intensity_keys(window)
    for tab in window.get_gui_experiment().get_all_tabs():
        experiment = tab.experiment
        for intensity_key in intensity_keys:
            intensity_calculator.remove_intensity_normalization(experiment, intensity_key=intensity_key)
        tab.undo_redo.mark_unsaved_changes()

