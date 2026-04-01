"""Allows you to assign a name to a channel."""
from typing import Optional, Dict, Any

from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageChannel
from organoid_tracker.core.images import ChannelDescription
from organoid_tracker.gui import dialog
from organoid_tracker.gui.undo_redo import UndoableAction
from organoid_tracker.gui.window import Window


class _ChangeChannelDescriptionAction(UndoableAction):
    """Action to change a channel name."""
    _channel: ImageChannel
    _old_name: str
    _new_name: str

    def __init__(self, channel: ImageChannel, old_name: str, new_name: str):
        self._channel = channel
        self._old_name = old_name
        self._new_name = new_name

    def do(self, experiment: Experiment) -> str:
        description = experiment.images.get_channel_description(self._channel)
        experiment.images.set_channel_description(self._channel, description.with_name(self._new_name))
        return f"Renamed channel to '{self._new_name}'."

    def undo(self, experiment: Experiment) -> str:
        description = experiment.images.get_channel_description(self._channel)
        experiment.images.set_channel_description(self._channel, description.with_name(self._old_name))
        return f"Renamed channel back to '{self._old_name}'."


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Edit//Experiment-Set channel name...": lambda: _set_channel_name(window)
    }


def _get_current_channel_name(window: Window) -> Optional[str]:
    """Returns the channel name for the currently selected channel, for the currently active experiments. If multiple
    experiments are active, and they differ in the name, None is returned. If the channel isn't available in any
    of the active experiments, None is returned as well."""
    image_channel = window.display_settings.image_channel

    image_channel_name = None
    for experiment in window.get_active_experiments():
        if image_channel.index_zero >= len(experiment.images.get_channels()):
            continue  # Channel name not available
        experiment_image_channel_name = experiment.images.get_channel_description(image_channel).channel_name
        if image_channel_name is not None and experiment_image_channel_name != image_channel_name:
            return None  # Not all open experiment names use the same name for the channel
        image_channel_name = experiment_image_channel_name
    return image_channel_name


def _set_channel_name(window: Window):
    """Prompts the user for a new channel name, and applies it."""
    image_channel = window.display_settings.image_channel
    current_name = _get_current_channel_name(window)
    if current_name is None:
        current_name = ""

    new_name = dialog.prompt_str("Channel name", f"What should be the new name for channel {image_channel.index_one}?", current_name)
    if new_name is None:
        return  # User pressed cancel

    experiment_count = 0
    skipped_count = 0
    for tab in window.get_gui_experiment().get_active_tabs():
        experiment_count += 1

        experiment = tab.experiment
        if image_channel.index_zero >= len(experiment.images.get_channels()):
            skipped_count += 1
            continue

        old_name = experiment.images.get_channel_description(image_channel).channel_name
        tab.undo_redo.do(_ChangeChannelDescriptionAction(image_channel, old_name, new_name), experiment)

    if experiment_count - skipped_count > 0:
        window.redraw_data()

    if experiment_count == 1 and skipped_count == 0:
        window.set_status(f"Set the name of channel {image_channel.index_one} to '{new_name}'.")
    elif experiment_count == skipped_count:
        window.set_status(f"Channel {image_channel.index_one} is not available in any of the active experiments, so no name was set.")
    elif skipped_count == 0:
        window.set_status(f"Set the name of channel {image_channel.index_one} to '{new_name}' for {experiment_count} experiments.")
    else:
        window.set_status(f"Set the name of channel {image_channel.index_one} to '{new_name}' for {experiment_count - skipped_count}"
                          f" experiment(s). For {skipped_count} experiment(s), the channel was not available.")